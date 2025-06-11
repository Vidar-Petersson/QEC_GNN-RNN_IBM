from utils_ibm import IBM_sampler

sampler = IBM_sampler(distance=3, t=5, batch_size=1000)
print("init")
detection_events, observable_flips = sampler.load_jobdata()


print(detection_events.shape)
print(observable_flips)