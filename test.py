import pyaudio


def get_input_device_id(device_name, microphones):
    for device in microphones:
        if device_name.lower() in device[1].lower():
            print(f"Selected device: {device[1]} (Index: {device[0]})")
            return device[0]
    if microphones:
        print(f"'{device_name}' not found. Using first available: {microphones[0][1]}")
        return microphones[0][0]
    return None


def list_microphones(pyaudio_instance):
    info = pyaudio_instance.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    result = []
    for i in range(0, numdevices):
        if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = pyaudio_instance.get_device_info_by_host_api_device_index(
                0, i).get('name')
            result += [[i, name]]
    return result


def check_device_info(audio, device_id):
    device_info = audio.get_device_info_by_index(device_id)
    print(f"Device info: {device_info}")
    return device_info


if __name__ == "__main__":
    audio = pyaudio.PyAudio()
    microphones = list_microphones(audio)
    selected_input_device_id = get_input_device_id('ATR4697-USB: USB Audio (hw:2,0)', microphones)
    device_info = check_device_info(audio, selected_input_device_id)