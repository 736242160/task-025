#!/usr/bin/env python3
"""circuit_breaker.py — 单文件熔断器命令行工具。

用法:
    python3 circuit_breaker.py demo   # 跑一个脚本化演示，观察状态切换
    python3 circuit_breaker.py test   # 运行内置测试用例

默认参数及理由:
    --failure-threshold 5     连续失败 5 次才熔断。偶发抖动(超时/丢包)很常见,
                              阈值太低会误伤;5 是业界常用默认值,既能容忍瞬时
                              故障,又能在服务真正挂掉时较快止损。
    --open-duration 30        熔断打开 30 秒。下游恢复通常需要数十秒(重启、
                              连接池排空、限流解除),30s 足够让多数故障自愈,
                              又不至于让用户等太久。
    --half-open-max-calls 3   半开时最多同时放 3 个试探请求。只放 1 个太脆
                              (单个成功可能是侥幸),放太多又可能把刚恢复的
                              服务再次打垮;3 个能在"验证恢复"和"保护下游"
                              之间取得平衡。任一试探成功即关闭,任一失败即
                              重新打开。
"""

from __future__ import annotations

import argparse
import enum
import threading
import time
import unittest


class CircuitOpenError(Exception):
    """熔断器处于打开(或半开且试探名额已满)状态时快速抛出。"""

    def __init__(self, state: str, retry_after: float):
        self.state = state
        self.retry_after = retry_after
        super().__init__(
            f"熔断器已打开(state={state}), 请求被快速拒绝, "
            f"{retry_after:.1f}s 后可重试"
        )


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """线程安全的熔断器。

    所有状态读取与迁移都在同一把锁内完成,因此:
    - 熔断打开的瞬间,之后才来检查状态的请求一定看到 OPEN,直接快速失败,
      不会漏到外部服务(之前已被放行的在途请求无法召回,这是熔断器的固有语义);
    - 半开状态下通过计数器原子地发放试探名额,超额的请求快速失败,
      不会同时放进一堆试探请求。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        open_duration: float = 30.0,
        half_open_max_calls: int = 3,
        clock=time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须 >= 1")
        if open_duration <= 0:
            raise ValueError("open_duration 必须 > 0")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls 必须 >= 1")
        self.failure_threshold = failure_threshold
        self.open_duration = open_duration
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock

        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_in_flight = 0

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def call(self, func, *args, **kwargs):
        """通过熔断器调用 func。熔断打开时抛 CircuitOpenError 快速失败。"""
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    # ---- 内部状态机(全部在锁内执行) ----

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state is State.OPEN
            and self._clock() - self._opened_at >= self.open_duration
        ):
            self._state = State.HALF_OPEN
            self._half_open_in_flight = 0

    def _before_call(self) -> None:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is State.OPEN:
                retry_after = self.open_duration - (self._clock() - self._opened_at)
                raise CircuitOpenError(State.OPEN.value, max(retry_after, 0.0))
            if self._state is State.HALF_OPEN:
                if self._half_open_in_flight >= self.half_open_max_calls:
                    raise CircuitOpenError(State.HALF_OPEN.value, 0.0)
                self._half_open_in_flight += 1
            # CLOSED: 直接放行

    def _on_success(self) -> None:
        with self._lock:
            if self._state is State.HALF_OPEN:
                # 试探成功: 恢复关闭
                self._state = State.CLOSED
                self._half_open_in_flight = 0
            self._consecutive_failures = 0

    def _on_failure(self) -> None:
        with self._lock:
            if self._state is State.HALF_OPEN:
                # 试探失败: 重新打开, 冷却期重新计时
                self._state = State.OPEN
                self._opened_at = self._clock()
                self._half_open_in_flight = 0
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = State.OPEN
                self._opened_at = self._clock()


class ScriptedService:
    """可控 mock 外部服务: 按预设脚本依次成功/失败。

    script 中每个元素:
        "ok"   -> 返回正常结果
        "fail" -> 抛出 RuntimeError
    脚本用完后默认成功。latency 模拟网络耗时。
    """

    def __init__(self, script, latency: float = 0.0):
        self._script = list(script)
        self._latency = latency
        self._lock = threading.Lock()
        self.call_count = 0

    def __call__(self):
        with self._lock:
            self.call_count += 1
            action = self._script.pop(0) if self._script else "ok"
        if self._latency:
            time.sleep(self._latency)
        if action == "fail":
            raise RuntimeError("外部服务故障(脚本预设)")
        return "ok"


# ---------------------------------------------------------------- 演示

def run_demo(args) -> None:
    # 脚本: 5 次失败 -> 触发熔断; 冷却期间脚本不被消费(快速失败);
    # 之后 1 次失败(半开试探失败, 重新打开); 再之后全部成功(恢复关闭)。
    script = ["fail"] * 5 + ["fail"] + ["ok"] * 10
    service = ScriptedService(script, latency=0.05)
    breaker = CircuitBreaker(
        failure_threshold=args.failure_threshold,
        open_duration=args.open_duration,
        half_open_max_calls=args.half_open_max_calls,
    )
    print(
        f"参数: failure_threshold={breaker.failure_threshold}, "
        f"open_duration={breaker.open_duration}s, "
        f"half_open_max_calls={breaker.half_open_max_calls}"
    )
    for i in range(1, 21):
        try:
            result = breaker.call(service)
            print(f"第{i:2d}次: 成功 -> {result!r}  [状态: {breaker.state.value}]")
        except CircuitOpenError as exc:
            print(f"第{i:2d}次: 快速失败 ({exc})  [状态: {breaker.state.value}]")
        except RuntimeError as exc:
            print(f"第{i:2d}次: 调用失败 ({exc})  [状态: {breaker.state.value}]")
        # 第 7、11、15 次之后多睡一会儿, 让冷却期过去以便观察半开切换
        time.sleep(0.4 if i in (7, 11, 15) else 0.1)
    print(f"外部服务实际被调用次数: {service.call_count}")


# ---------------------------------------------------------------- 测试

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CircuitBreakerTest(unittest.TestCase):
    def make_breaker(self, **overrides):
        clock = FakeClock()
        params = dict(
            failure_threshold=3,
            open_duration=10.0,
            half_open_max_calls=2,
            clock=clock,
        )
        params.update(overrides)
        return CircuitBreaker(**params), clock

    def test_consecutive_failures_open_circuit(self):
        """连续失败达到阈值后熔断打开。"""
        breaker, _ = self.make_breaker()
        service = ScriptedService(["fail"] * 3)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                breaker.call(service)
        self.assertIs(breaker.state, State.CLOSED)  # 未到阈值仍关闭
        with self.assertRaises(RuntimeError):
            breaker.call(service)
        self.assertIs(breaker.state, State.OPEN)  # 第 3 次失败触发熔断

    def test_open_state_fails_fast_without_calling_service(self):
        """打开期间请求快速失败, 不真正调用外部服务。"""
        breaker, _ = self.make_breaker()
        service = ScriptedService(["fail"] * 3 + ["ok"])
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                breaker.call(service)
        self.assertIs(breaker.state, State.OPEN)
        calls_before = service.call_count
        for _ in range(5):
            with self.assertRaises(CircuitOpenError):
                breaker.call(service)
        self.assertEqual(service.call_count, calls_before)  # 一次都没打过去

    def test_half_open_probe_success_closes_circuit(self):
        """冷却结束后进入半开, 试探成功则恢复关闭。"""
        breaker, clock = self.make_breaker()
        service = ScriptedService(["fail"] * 3 + ["ok", "ok"])
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                breaker.call(service)
        clock.advance(10.0)  # 冷却期结束
        self.assertIs(breaker.state, State.HALF_OPEN)
        self.assertEqual(breaker.call(service), "ok")  # 试探成功
        self.assertIs(breaker.state, State.CLOSED)
        self.assertEqual(breaker.call(service), "ok")  # 后续正常放行

    def test_half_open_probe_failure_reopens_circuit(self):
        """半开试探失败则重新打开, 冷却期重新计时。"""
        breaker, clock = self.make_breaker()
        service = ScriptedService(["fail"] * 4)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                breaker.call(service)
        clock.advance(10.0)
        with self.assertRaises(RuntimeError):
            breaker.call(service)  # 半开试探失败
        self.assertIs(breaker.state, State.OPEN)
        # 旧的冷却时间点不能再触发半开
        clock.advance(9.9)
        with self.assertRaises(CircuitOpenError):
            breaker.call(service)
        clock.advance(0.1)  # 新冷却期结束
        self.assertIs(breaker.state, State.HALF_OPEN)

    def test_half_open_limits_concurrent_probes(self):
        """半开状态下并发请求只放行限定数量的试探, 其余快速失败。"""
        breaker, clock = self.make_breaker(half_open_max_calls=2)
        service = ScriptedService(["fail"] * 3)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                breaker.call(service)
        clock.advance(10.0)

        entered = threading.Barrier(3)  # 2 个试探线程 + 主线程
        release = threading.Event()
        results = []

        def slow_service():
            entered.wait(timeout=5)   # 保证两个试探都在飞行中
            release.wait(timeout=5)
            return "ok"

        def probe():
            try:
                results.append(breaker.call(slow_service))
            except Exception as exc:  # pragma: no cover
                results.append(exc)

        threads = [threading.Thread(target=probe) for _ in range(2)]
        for t in threads:
            t.start()
        entered.wait(timeout=5)  # 两个试探都已进入外部调用
        # 此时半开名额已满, 新请求必须快速失败且不能调用服务
        with self.assertRaises(CircuitOpenError):
            breaker.call(lambda: "should-not-run")
        release.set()
        for t in threads:
            t.join()
        self.assertEqual(results, ["ok", "ok"])
        self.assertIs(breaker.state, State.CLOSED)

    def test_no_leak_at_the_moment_circuit_opens(self):
        """熔断刚打开的一瞬间, 之后到达的请求不能漏到外部服务。"""
        breaker, _ = self.make_breaker(failure_threshold=1)
        hits = []

        def failing_service():
            hits.append(1)
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            breaker.call(failing_service)
        self.assertIs(breaker.state, State.OPEN)
        # 打开瞬间之后并发涌入的请求全部快速失败
        def worker():
            try:
                breaker.call(failing_service)
            except CircuitOpenError:
                pass
        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(hits), 1)  # 只有触发熔断的那一次真正调用了服务


def run_tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CircuitBreakerTest)
    runner = unittest.TextTestRunner(verbosity=2)
    raise SystemExit(0 if runner.run(suite).wasSuccessful() else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="熔断器命令行工具")
    parser.add_argument(
        "--failure-threshold", type=int, default=5,
        help="连续失败多少次后熔断打开 (默认: 5)",
    )
    parser.add_argument(
        "--open-duration", type=float, default=30.0,
        help="熔断打开持续秒数, 之后进入半开 (默认: 30)",
    )
    parser.add_argument(
        "--half-open-max-calls", type=int, default=3,
        help="半开状态下最多同时放行几个试探请求 (默认: 3)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="运行脚本化演示")
    sub.add_parser("test", help="运行内置测试")
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args)
    else:
        run_tests()


if __name__ == "__main__":
    main()
