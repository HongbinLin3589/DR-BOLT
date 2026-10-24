"""Wall time and per-GPU resident memory for upstream subprocess trees."""

import subprocess
import time


def measured_command(command, *, cwd, env):
    import psutil
    import pynvml

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    peaks = [None] * len(handles)
    samples = [0] * len(handles)
    unavailable = [0] * len(handles)
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, env=env)
    tracked = {process.pid}
    try:
        while process.poll() is None:
            try:
                tracked.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            for index, handle in enumerate(handles):
                processes = [item for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                             if item.pid in tracked]
                if not processes:
                    continue
                if any(item.usedGpuMemory == pynvml.NVML_VALUE_NOT_AVAILABLE or item.usedGpuMemory is None for item in processes):
                    unavailable[index] += 1
                    continue
                resident = sum(item.usedGpuMemory for item in processes)
                samples[index] += 1
                peaks[index] = resident if peaks[index] is None else max(peaks[index], resident)
            time.sleep(0.25)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        pynvml.nvmlShutdown()
    observed = [value for value in peaks if value is not None]
    result = {"seconds": time.perf_counter() - start, "per_gpu_peak_bytes": peaks,
              "valid_memory_samples": samples, "unavailable_memory_samples": unavailable,
              "memory_status": "unavailable" if not observed else "partial" if any(unavailable) else "measured",
              "memory_measurement": "NVML process-tree resident bytes"}
    if observed:
        result["peak_gpu_bytes"] = max(observed)
    else:
        result["memory_reason"] = "No valid process-tree GPU memory observation was collected"
    return result
