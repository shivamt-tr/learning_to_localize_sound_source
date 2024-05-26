import pynvml

def get_gpu_temperature():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    temperatures = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        temperatures.append(temperature)

    pynvml.nvmlShutdown()
    return temperatures

if __name__ == "__main__":
    temperatures = get_gpu_temperature()
    for idx, temp in enumerate(temperatures):
        print(f"GPU {idx} Temperature: {temp}°C")
