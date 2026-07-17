# gemma4 probe — 20260420-085855

- container: `vllm_gemma4-31b`
- model: `cyankiwi/gemma-4-31B-it-AWQ-4bit`
- host: http://localhost:8000
- engine core pid (at start): 422

## Initial state
```
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
31104 MiB, 32607 MiB, 0 %
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
first: 2026/04/20 08:58:55.387, 31104, 32607, 0, 22, 15.41, 29
peak:  2026/04/20 08:58:55.387, 31104, 32607, 0, 22, 15.41, 29
last:  2026/04/20 08:59:54.811, 31104, 32607, 0, 21, 15.38, 30
```

### Stack traces (captured because hang detected)
```
=== container ps ===
    PID    PPID COMMAND         STAT %CPU %MEM
      1       0 vllm            Ssl   2.7  2.3
    421       1 python          S     0.0  0.0
    422       1 VLLM::EngineCor Sl    5.7  4.1
   3772       0 ps              Rs   33.3  0.0

=== APIServer pid=1 /proc/1/stack ===
(unavailable)

=== EngineCore pid=422 /proc/422/stack ===
(unavailable)

=== EngineCore task stacks ===
--- /proc/422/task/1090/stack ---
(unavailable)
--- /proc/422/task/1707/stack ---
(unavailable)
--- /proc/422/task/1708/stack ---
(unavailable)
--- /proc/422/task/1709/stack ---
(unavailable)
--- /proc/422/task/1710/stack ---
(unavailable)
--- /proc/422/task/1711/stack ---
(unavailable)
--- /proc/422/task/1712/stack ---
(unavailable)
--- /proc/422/task/1713/stack ---
(unavailable)
--- /proc/422/task/422/stack ---
(unavailable)
--- /proc/422/task/438/stack ---
(unavailable)
--- /proc/422/task/439/stack ---
(unavailable)
--- /proc/422/task/440/stack ---
(unavailable)
--- /proc/422/task/441/stack ---
(unavailable)
--- /proc/422/task/442/stack ---
(unavailable)
--- /proc/422/task/443/stack ---
(unavailable)
--- /proc/422/task/444/stack ---
(unavailable)
--- /proc/422/task/445/stack ---
(unavailable)
--- /proc/422/task/446/stack ---
(unavailable)
--- /proc/422/task/447/stack ---
(unavailable)
--- /proc/422/task/448/stack ---
(unavailable)
--- /proc/422/task/449/stack ---
(unavailable)
--- /proc/422/task/450/stack ---
(unavailable)
--- /proc/422/task/451/stack ---
(unavailable)
--- /proc/422/task/452/stack ---
(unavailable)
--- /proc/422/task/453/stack ---
(unavailable)
--- /proc/422/task/454/stack ---
(unavailable)
--- /proc/422/task/455/stack ---
(unavailable)
--- /proc/422/task/456/stack ---
(unavailable)
--- /proc/422/task/457/stack ---
(unavailable)
--- /proc/422/task/458/stack ---
(unavailable)
--- /proc/422/task/459/stack ---
(unavailable)
--- /proc/422/task/460/stack ---
(unavailable)
--- /proc/422/task/468/stack ---
(unavailable)
--- /proc/422/task/491/stack ---
(unavailable)
--- /proc/422/task/492/stack ---
(unavailable)
--- /proc/422/task/493/stack ---
(unavailable)
--- /proc/422/task/494/stack ---
(unavailable)
--- /proc/422/task/495/stack ---
(unavailable)
--- /proc/422/task/496/stack ---
(unavailable)
--- /proc/422/task/497/stack ---
(unavailable)
--- /proc/422/task/498/stack ---
(unavailable)
--- /proc/422/task/499/stack ---
(unavailable)
--- /proc/422/task/500/stack ---
(unavailable)
--- /proc/422/task/501/stack ---
(unavailable)
--- /proc/422/task/502/stack ---
(unavailable)
--- /proc/422/task/503/stack ---
(unavailable)
--- /proc/422/task/504/stack ---
(unavailable)
--- /proc/422/task/505/stack ---
(unavailable)
--- /proc/422/task/506/stack ---
(unavailable)
--- /proc/422/task/507/stack ---
(unavailable)
--- /proc/422/task/508/stack ---
(unavailable)
--- /proc/422/task/509/stack ---
(unavailable)
--- /proc/422/task/510/stack ---
(unavailable)
--- /proc/422/task/511/stack ---
(unavailable)
--- /proc/422/task/512/stack ---
(unavailable)
--- /proc/422/task/513/stack ---
(unavailable)
--- /proc/422/task/521/stack ---
(unavailable)
--- /proc/422/task/522/stack ---
(unavailable)
--- /proc/422/task/523/stack ---
(unavailable)
--- /proc/422/task/524/stack ---
(unavailable)
--- /proc/422/task/525/stack ---
(unavailable)
--- /proc/422/task/526/stack ---
(unavailable)
--- /proc/422/task/527/stack ---
(unavailable)
--- /proc/422/task/528/stack ---
(unavailable)
--- /proc/422/task/529/stack ---
(unavailable)
--- /proc/422/task/530/stack ---
(unavailable)
--- /proc/422/task/531/stack ---
(unavailable)
--- /proc/422/task/532/stack ---
(unavailable)
--- /proc/422/task/533/stack ---
(unavailable)
--- /proc/422/task/534/stack ---
(unavailable)
--- /proc/422/task/535/stack ---
(unavailable)
--- /proc/422/task/536/stack ---
(unavailable)
--- /proc/422/task/537/stack ---
(unavailable)
--- /proc/422/task/538/stack ---
(unavailable)
--- /proc/422/task/539/stack ---
(unavailable)
--- /proc/422/task/540/stack ---
(unavailable)
--- /proc/422/task/541/stack ---
(unavailable)
--- /proc/422/task/542/stack ---
(unavailable)
--- /proc/422/task/543/stack ---
(unavailable)
--- /proc/422/task/605/stack ---
(unavailable)
--- /proc/422/task/606/stack ---
(unavailable)
--- /proc/422/task/607/stack ---
(unavailable)
--- /proc/422/task/608/stack ---
(unavailable)
--- /proc/422/task/609/stack ---
(unavailable)
--- /proc/422/task/610/stack ---
(unavailable)
--- /proc/422/task/611/stack ---
(unavailable)
--- /proc/422/task/612/stack ---
(unavailable)
--- /proc/422/task/613/stack ---
(unavailable)
--- /proc/422/task/614/stack ---
(unavailable)
--- /proc/422/task/615/stack ---
(unavailable)
--- /proc/422/task/616/stack ---
(unavailable)
--- /proc/422/task/617/stack ---
(unavailable)
--- /proc/422/task/618/stack ---
(unavailable)
```

## Aborted — engine hung on `simple`

Remaining tests skipped. Restart container with `vllm rec 4` and re-run.
