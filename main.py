from models.tensors import Tensors
from iPEPS import iPEPS

import pickle
import torch
import numpy as np
import os


# Run the model on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

lambdas = [lam for lam in np.arange(0.1, 4.5, 0.25)]
data = {"lambdas": lambdas, "energies": [], "Mx": [], "My": [], "Mz": [], "Mg": []}

Mpx = Tensors.Mpx().to(device)
Mpy = Tensors.Mpy().to(device)
Mpz = Tensors.Mpz().to(device)

for lam in lambdas:
    success = False
    while not success:
        try:
            print("\nlambda:", lam)
            H = Tensors.H(lam=lam).to(device)

            model = iPEPS(chi=16, d=2, H=H, Mpx=Mpx, Mpy=Mpy, Mpz=Mpz).to(device)

            optimizer = torch.optim.LBFGS(model.parameters())

            def train():
                """
                Train the parameters of the iPEPS model.
                """
                optimizer.zero_grad()
                loss, Mx, My, Mz = model.forward()
                loss.backward()
                return loss

            for epoch in range(50):
                loss = optimizer.step(train)
                print("epoch:", epoch, "energy:", loss.item())

            with torch.no_grad():
                E, Mx, My, Mz = model.forward()
                data["energies"].append(E)
                data["Mx"].append(Mx)
                data["My"].append(My)
                data["Mz"].append(Mz)
                data["Mg"].append(torch.sqrt(Mx**2 + My**2 + Mz**2))
            success = True

        except torch._C._LinAlgError:
            print("LinAlgError occurred. Retrying...")


# Check if the file already exists and modify the filename if necessary
filename = "data/data.pkl"
if os.path.exists(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    filename = f"{base}_{counter}{ext}"

with open(filename, "wb") as f:
    pickle.dump(data, f)
