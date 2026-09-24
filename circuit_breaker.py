#!/usr/bin/env python3
"""单文件熔断器工具：熔断器实现 + 可控 mock 外部服务 + CLI 演示 + 测试。

用法：
    python3 circuit_breaker.py demo          # 跑一个完整的熔断演示
    python3 circuit_breaker.py test          # 运行全部测试用例
    python3 circuit_breaker.py demo --failure-threshold 3 --open-duration 1.0
"""

import argparse
import sys
import threading
import time
import unittest
from collections import deque

# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """熔断器处于打开（或半开且试探名额已满）时，请求快速失败抛出此异常。"""


class CircuitBreaker:
    """线程安全熔断器。

    参数默认值及理由：
      failure_threshold=5   连续失败 5 次才熔断。太低（如 1~2 次）会被偶发
                            网络抖动误触发；太高则故障期间放行太多无用请求、
                            拖垮线程池。5 是业界常见折中。
      open_duration=30.0    打开 30 秒后再试探。要大于外部服务典型的恢复
                            时间（重启、限流窗口通常是几十秒级），太短会
                            反复空转试探，太长则恢复后迟迟不可用。
      half_open_max_calls=1 半开时只放 1 个试探请求。刚恢复的服务最脆弱，
                            放多个并发试探容易把它再次打垮（惊群效应）；
                            1 个足以判断服务是否恢复，代价最小。
    """

    def __init__(self, failure_threshold=5, open_duration=30.0,
                 half_open_max_calls=1, clock=time.monotonic, name="breaker"):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        self.failure_threshold = failure_threshold
        self.open_duration = open_duration
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        self._clock = clock  # 可注入时钟，测试用假时钟保证确定性
        self._lock = threading.Lock()
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_in_flight = 0

    @property
    def state(self):
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def call(self, func, *args, **kwargs):
        """通过熔断器调用 func。打开期间直接抛 CircuitBreakerOpenError。"""
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    # -- 内部状态机（全部在锁内完成判断与计数，保证并发切换不出错） --

    def _maybe_transition_to_half_open(self):
        """调用方必须已持有锁。"""
        if (self._state == OPEN
                and self._clock() - self._opened_at >= self.open_duration):
            self._state = HALF_OPEN
            self._half_open_in_flight = 0

    def _before_call(self):
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == OPEN:
                remaining = self.open_duration - (self._clock() - self._opened_at)
                raise CircuitBreakerOpenError(
                    f"[{self.name}] 熔断器打开，快速失败（{remaining:.1f}s 后进入半开）")
            if self._state == HALF_OPEN:
                # 检查名额和占坑在同一个锁内完成：不会同时放进超额试探请求
                if self._half_open_in_flight >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"[{self.name}] 半开状态试探名额已满，快速失败")
                self._half_open_in_flight += 1

    def _on_success(self):
        with self._lock:
            if self._state == HALF_OPEN:
                self._half_open_in_flight -= 1
                self._close_locked()
            else:
                self._consecutive_failures = 0

    def _on_failure(self):
        with self._lock:
            if self._state == HALF_OPEN:
                self._half_open_in_flight -= 1
                self._open_locked()  # 试探失败：继续保持打开，重新计时
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._open_locked()

    def _open_locked(self):
        self._state = OPEN
        self._opened_at = self._clock()
        self._consecutive_failures = 0

    def _close_locked(self):
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_in_flight = 0


# ---------------------------------------------------------------------------
# 可控 mock 外部服务：按预设脚本成功或失败
# ---------------------------------------------------------------------------

class ExternalServiceError(RuntimeError):
    pass


class ScriptedService:
    """按脚本行动的假外部服务。脚本元素："ok" 或 "fail"，脚本用完后默认 "ok"。"""

    def __init__(self, script=(), latency=0.0):
        self._script = deque(script)
        self._latency = latency
        self._lock = threading.Lock()
        self.calls = 0  # 真实打到"外部服务"的次数，用于验证快速失败没漏网

    def __call__(self):
        with self._lock:
            self.calls += 1
            action = self._script.popleft() if self._script else "ok"
        if self._latency:
            time.sleep(self._latency)
        if action == "fail":
            raise ExternalServiceError("外部服务故障（脚本预设）")
        return "外部服务响应 OK"


# ---------------------------------------------------------------------------
# CLI 演示
# ---------------------------------------------------------------------------

def run_demo(args):
    service = ScriptedService(
        script=["fail"] * args.failure_threshold  # 先把服务打挂，触发熔断
               + ["fail"]                         # 半开第一次试探仍失败 -> 保持打开
               + ["ok"],                          # 之后服务恢复
    )
    breaker = CircuitBreaker(
        failure_threshold=args.failure_threshold,
        open_duration=args.open_duration,
        half_open_max_calls=args.half_open_max_calls,
        name="demo",
    )

    def attempt(tag):
        try:
            result = breaker.call(service)
            print(f"  [{breaker.state:9s}] {tag}: 成功 -> {result}")
        except CircuitBreakerOpenError as exc:
            print(f"  [{breaker.state:9s}] {tag}: 快速失败（未调用外部服务）{exc}")
        except ExternalServiceError as exc:
            print(f"  [{breaker.state:9s}] {tag}: 调用失败 -> {exc}")

    print(f"参数: failure_threshold={breaker.failure_threshold}, "
          f"open_duration={breaker.open_duration}s, "
          f"half_open_max_calls={breaker.half_open_max_calls}\n")

    print("阶段 1: 外部服务持续故障，连续失败触发熔断")
    for i in range(args.failure_threshold + 2):
        attempt(f"请求 {i + 1}")

    print(f"\n阶段 2: 熔断打开期间，请求全部快速失败（当前真实调用数={service.calls}）")
    for i in range(3):
        attempt(f"请求 {i + 1}")

    print(f"\n阶段 3: 等待 {args.open_duration}s 进入半开，第一次试探仍失败 -> 保持打开")
    time.sleep(args.open_duration)
    attempt("试探请求 1")
    attempt("请求（仍打开）")

    print(f"\n阶段 4: 再等 {args.open_duration}s，服务已恢复，试探成功 -> 恢复关闭")
    time.sleep(args.open_duration)
    attempt("试探请求 2")
    attempt("后续请求")

    print(f"\n外部服务真实被调用次数: {service.calls}")


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CircuitBreakerTest(unittest.TestCase):
    def make(self, script=(), threshold=3, open_duration=10.0, half_open_max=1):
        clock = FakeClock()
        service = ScriptedService(script)
        breaker = CircuitBreaker(failure_threshold=threshold,
                                 open_duration=open_duration,
                                 half_open_max_calls=half_open_max,
                                 clock=clock)
        return breaker, service, clock

    def test_consecutive_failures_trip_open(self):
        breaker, service, _ = self.make(script=["fail"] * 3, threshold=3)
        for _ in range(3):
            with self.assertRaises(ExternalServiceError):
                breaker.call(service)
        self.assertEqual(breaker.state, OPEN)
        self.assertEqual(service.calls, 3)

    def test_open_state_fails_fast_without_calling_service(self):
        breaker, service, _ = self.make(script=["fail"] * 3, threshold=3)
        for _ in range(3):
            with self.assertRaises(ExternalServiceError):
                breaker.call(service)
        calls_before = service.calls
        for _ in range(5):
            with self.assertRaises(CircuitBreakerOpenError):
                breaker.call(service)
        self.assertEqual(service.calls, calls_before)  # 一个漏网请求都没有

    def test_half_open_probe_success_closes(self):
        breaker, service, clock = self.make(script=["fail"] * 3 + ["ok"], threshold=3)
        for _ in range(3):
            with self.assertRaises(ExternalServiceError):
                breaker.call(service)
        clock.advance(10.0)  # 超过 open_duration，进入半开
        result = breaker.call(service)  # 试探成功
        self.assertEqual(result, "外部服务响应 OK")
        self.assertEqual(breaker.state, CLOSED)
        # 恢复后正常流量放行
        breaker.call(service)
        self.assertEqual(breaker.state, CLOSED)

    def test_half_open_probe_failure_stays_open(self):
        breaker, service, clock = self.make(script=["fail"] * 4, threshold=3)
        for _ in range(3):
            with self.assertRaises(ExternalServiceError):
                breaker.call(service)
        clock.advance(10.0)
        with self.assertRaises(ExternalServiceError):
            breaker.call(service)  # 半开试探失败
        self.assertEqual(breaker.state, OPEN)
        # 打开窗口重新计时：再推进不足一个窗口仍快速失败
        clock.advance(9.9)
        with self.assertRaises(CircuitBreakerOpenError):
            breaker.call(service)
        self.assertEqual(service.calls, 4)

    def test_half_open_limits_concurrent_probes(self):
        """半开状态并发抢试探名额：只有 1 个能进去，其余快速失败。"""
        clock = FakeClock()
        entered = threading.Barrier(2)  # 主线程 + 试探线程同步
        release = threading.Event()

        def slow_service():
            entered.wait(timeout=5)   # 等主线程也到位，制造并发窗口
            release.wait(timeout=5)   # 占住试探名额不放手
            return "ok"

        breaker = CircuitBreaker(failure_threshold=1, open_duration=10.0,
                                 half_open_max_calls=1, clock=clock)
        with self.assertRaises(ExternalServiceError):
            breaker.call(ScriptedService(["fail"]))
        clock.advance(10.0)

        outcome = {}
        probe = threading.Thread(
            target=lambda: outcome.setdefault("probe", breaker.call(slow_service)))
        probe.start()
        entered.wait(timeout=5)  # 确认试探线程已占住名额

        # 此时并发请求必须快速失败，不能成为第二个试探
        with self.assertRaises(CircuitBreakerOpenError):
            breaker.call(slow_service)

        release.set()
        probe.join(timeout=5)
        self.assertEqual(outcome["probe"], "ok")
        self.assertEqual(breaker.state, CLOSED)

    def test_open_moment_no_slip_through(self):
        """熔断刚打开的一瞬间，并发请求不能漏到外部服务。"""
        clock = FakeClock()
        service = ScriptedService(["fail"] * 3)
        breaker = CircuitBreaker(failure_threshold=3, open_duration=10.0, clock=clock)
        for _ in range(3):
            with self.assertRaises(ExternalServiceError):
                breaker.call(service)
        calls_before = service.calls
        errors = []

        def hammer():
            try:
                breaker.call(service)
            except CircuitBreakerOpenError:
                errors.append("fast-fail")
            except ExternalServiceError:
                errors.append("called-service")

        threads = [threading.Thread(target=hammer) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(service.calls, calls_before)  # 没有任何请求漏进去
        self.assertEqual(set(errors), {"fast-fail"})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="运行熔断演示")
    demo.add_argument("--failure-threshold", type=int, default=3)
    demo.add_argument("--open-duration", type=float, default=2.0)
    demo.add_argument("--half-open-max-calls", type=int, default=1)
    sub.add_parser("test", help="运行测试")
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args)
    else:
        unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    main()
