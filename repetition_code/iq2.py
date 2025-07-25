from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder, SamplerV2 as Sampler, Batch
import os
from qiskit_aer import AerSimulator 
import numpy as np
import json

# 1) Ladda din IBMQ‑konto och välj en backend
service = QiskitRuntimeService()
backend = service.backend("ibm_kingston")  # Specify backend

# 3. Skapa kalibreringskretsar |0⟩ och |1⟩
calib_qcs = []
for state in ['0', '1']:
    qc = QuantumCircuit(1, 1, name=f"calib_{state}")
    if state == '1':
        qc.x(0)
    qc.measure_all()
    qc.metadata = {'label': state}
    calib_qcs.append(qc)

# 4. Skapa din huvudkrets (ex: Hadamard-experiment)
main_qc = QuantumCircuit(1, 1, name="experiment")
main_qc.h(0)
main_qc.measure_all()
main_qc.metadata = {'label': 'experiment'}

all_qcs = calib_qcs + [main_qc]

# 5. Transpilera för vald backend
transpiled = transpile(all_qcs, backend=backend, optimization_level=1)

# 6. Skapa SamplerV2 med kerneled IQ-mätning (meas_level = 1)
sampler = Sampler(
    mode=backend,
    options={
        "default_shots": 1024,
        "execution": {
            "meas_type": "kerneled"  # Detta ger IQ-värden per skott
        }
    }
)

# 7. Kör alla kretsar med SamplerV2
pubs = [(qc,) for qc in transpiled]
job = sampler.run(pubs)
print("Job submitted:", job.job_id())

# 8. Vänta in resultatet
res = job.result()

# 8) Spara hela resultatobjektet som JSON
output_dir = "./jobdata/ibm/iq_data"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(
    output_dir,
    f"{job.job_id()}_iq_results.json"
)

with open(filename, "w") as f:
    json.dump(res, f, cls=RuntimeEncoder)

print(f"Measurement saved as '{filename}'.")

# 9) Extrahera och konvertera IQ‑data
iq_data = {}
for pub_res, qc in zip(res, all_qcs):
    label = qc.metadata["label"]
    raw = pub_res.data.kerneled  # shape = (shots, 2), float
    arr = np.asarray(raw, dtype=float)
    complex_iq = arr[:, 0] + 1j * arr[:, 1]
    iq_data[label] = complex_iq

# 10) Visa exempel
for label, data in iq_data.items():
    print(f"\nTillstånd |{label}⟩, första 10 IQ‑punkter:")
    for μ in data[:10]:
        print(f"  {μ.real:.3f} + {μ.imag:.3f}j")