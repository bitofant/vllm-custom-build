# gemma4 probe — 20260419-221901

- container: `vllm_gemma4-31b`
- model: `cyankiwi/gemma-4-31B-it-AWQ-4bit`
- host: http://localhost:8000
- engine core pid (at start): 435

## Initial state
```
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
31110 MiB, 32607 MiB, 0 %
```

## Test: `simple` — Short chat, no tools, non-stream

- elapsed: **60.01s**
- curl_exit: 28
- http_code: ?
- bytes: 0
- TTFB: ?s
- hung: **YES**

### GPU samples (first/peak/last)
```
timestamp,mem_used_MiB,mem_total_MiB,gpu_util,mem_util,power_W,temp_C
first: 2026/04/19 22:19:02.031, 31110, 32607, 0, 38, 15.42, 33
peak:  2026/04/19 22:19:02.031, 31110, 32607, 0, 38, 15.42, 33
last:  2026/04/19 22:20:01.485, 31110, 32607, 0, 33, 15.40, 33
```

### Stack traces (captured because hang detected)
```
=== container ps ===
    PID    PPID COMMAND         STAT %CPU %MEM
      1       0 vllm            Ssl   1.4  2.3
    434       1 python          S     0.0  0.0
    435       1 VLLM::EngineCor Sl    7.7  4.1
   2538       0 ps              Rs    0.0  0.0

=== APIServer pid=1 /proc/1/stack ===
(unavailable)

=== EngineCore pid=435 /proc/435/stack ===
(unavailable)

=== EngineCore task stacks ===
--- /proc/435/task/1110/stack ---
(unavailable)
--- /proc/435/task/1705/stack ---
(unavailable)
--- /proc/435/task/1706/stack ---
(unavailable)
--- /proc/435/task/1707/stack ---
(unavailable)
--- /proc/435/task/1708/stack ---
(unavailable)
--- /proc/435/task/1709/stack ---
(unavailable)
--- /proc/435/task/1710/stack ---
(unavailable)
--- /proc/435/task/1711/stack ---
(unavailable)
--- /proc/435/task/1837/stack ---
(unavailable)
--- /proc/435/task/435/stack ---
(unavailable)
--- /proc/435/task/443/stack ---
(unavailable)
--- /proc/435/task/444/stack ---
(unavailable)
--- /proc/435/task/445/stack ---
(unavailable)
--- /proc/435/task/446/stack ---
(unavailable)
--- /proc/435/task/447/stack ---
(unavailable)
--- /proc/435/task/448/stack ---
(unavailable)
--- /proc/435/task/449/stack ---
(unavailable)
--- /proc/435/task/450/stack ---
(unavailable)
--- /proc/435/task/451/stack ---
(unavailable)
--- /proc/435/task/452/stack ---
(unavailable)
--- /proc/435/task/453/stack ---
(unavailable)
--- /proc/435/task/454/stack ---
(unavailable)
--- /proc/435/task/455/stack ---
(unavailable)
--- /proc/435/task/456/stack ---
(unavailable)
--- /proc/435/task/457/stack ---
(unavailable)
--- /proc/435/task/458/stack ---
(unavailable)
--- /proc/435/task/459/stack ---
(unavailable)
--- /proc/435/task/460/stack ---
(unavailable)
--- /proc/435/task/461/stack ---
(unavailable)
--- /proc/435/task/462/stack ---
(unavailable)
--- /proc/435/task/463/stack ---
(unavailable)
--- /proc/435/task/464/stack ---
(unavailable)
--- /proc/435/task/465/stack ---
(unavailable)
--- /proc/435/task/466/stack ---
(unavailable)
--- /proc/435/task/488/stack ---
(unavailable)
--- /proc/435/task/489/stack ---
(unavailable)
--- /proc/435/task/490/stack ---
(unavailable)
--- /proc/435/task/491/stack ---
(unavailable)
--- /proc/435/task/492/stack ---
(unavailable)
--- /proc/435/task/493/stack ---
(unavailable)
--- /proc/435/task/494/stack ---
(unavailable)
--- /proc/435/task/495/stack ---
(unavailable)
--- /proc/435/task/496/stack ---
(unavailable)
--- /proc/435/task/497/stack ---
(unavailable)
--- /proc/435/task/498/stack ---
(unavailable)
--- /proc/435/task/499/stack ---
(unavailable)
--- /proc/435/task/500/stack ---
(unavailable)
--- /proc/435/task/501/stack ---
(unavailable)
--- /proc/435/task/502/stack ---
(unavailable)
--- /proc/435/task/503/stack ---
(unavailable)
--- /proc/435/task/504/stack ---
(unavailable)
--- /proc/435/task/505/stack ---
(unavailable)
--- /proc/435/task/506/stack ---
(unavailable)
--- /proc/435/task/507/stack ---
(unavailable)
--- /proc/435/task/508/stack ---
(unavailable)
--- /proc/435/task/509/stack ---
(unavailable)
--- /proc/435/task/510/stack ---
(unavailable)
--- /proc/435/task/533/stack ---
(unavailable)
--- /proc/435/task/534/stack ---
(unavailable)
--- /proc/435/task/535/stack ---
(unavailable)
--- /proc/435/task/536/stack ---
(unavailable)
--- /proc/435/task/537/stack ---
(unavailable)
--- /proc/435/task/538/stack ---
(unavailable)
--- /proc/435/task/539/stack ---
(unavailable)
--- /proc/435/task/540/stack ---
(unavailable)
--- /proc/435/task/541/stack ---
(unavailable)
--- /proc/435/task/542/stack ---
(unavailable)
--- /proc/435/task/543/stack ---
(unavailable)
--- /proc/435/task/544/stack ---
(unavailable)
--- /proc/435/task/545/stack ---
(unavailable)
--- /proc/435/task/546/stack ---
(unavailable)
--- /proc/435/task/547/stack ---
(unavailable)
--- /proc/435/task/548/stack ---
(unavailable)
--- /proc/435/task/549/stack ---
(unavailable)
--- /proc/435/task/550/stack ---
(unavailable)
--- /proc/435/task/551/stack ---
(unavailable)
--- /proc/435/task/552/stack ---
(unavailable)
--- /proc/435/task/553/stack ---
(unavailable)
--- /proc/435/task/554/stack ---
(unavailable)
--- /proc/435/task/555/stack ---
(unavailable)
--- /proc/435/task/603/stack ---
(unavailable)
--- /proc/435/task/604/stack ---
(unavailable)
--- /proc/435/task/605/stack ---
(unavailable)
--- /proc/435/task/606/stack ---
(unavailable)
--- /proc/435/task/607/stack ---
(unavailable)
--- /proc/435/task/608/stack ---
(unavailable)
--- /proc/435/task/609/stack ---
(unavailable)
--- /proc/435/task/610/stack ---
(unavailable)
--- /proc/435/task/611/stack ---
(unavailable)
--- /proc/435/task/612/stack ---
(unavailable)
--- /proc/435/task/613/stack ---
(unavailable)
--- /proc/435/task/614/stack ---
(unavailable)
--- /proc/435/task/615/stack ---
(unavailable)
```

## Aborted — engine hung on `simple`

Remaining tests skipped. Restart container with `vllm rec 4` and re-run.
