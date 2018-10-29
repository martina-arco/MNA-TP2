def filter(fft, frequencies):
    print("HOLA")
    result = []
    for i in range(len(frequencies)):
        if 40 <= frequencies[i] <= 120:
            result.append(fft[i])
        else:
            result.append(0)

    return result