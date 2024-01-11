# Nu træner vi kun 10 gange på første billede. 
# Du skal lave et forloop som går gennem alle billeder og tester 100 gange på alle billeder

import Andreas_aotu.y_k_means_funk as y_k
import torch

def Hele_encoder (x1):
    N, D_picture, H, D_bottle_neck = 64, len(y_k.lav_y_k_means("1.jpg")), 100, 10

    class Aotu_encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(D_picture, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_bottle_neck),
            )

            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(D_bottle_neck, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_picture),
            )

        def forward (self, billed):
            billed_enc = self.encoder(billed)
            billed_dec = self.decoder(billed_enc)
            return billed_dec 

    model = Aotu_encoder()

    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    for i1 in range(1,x1):
        for i2 in range(50):
            y_pred = model(y_k.lav_y_k_means(f"{i1}.jpg"))

            loss = loss_fn(y_pred, y_k.lav_y_k_means("1.jpg"))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


        y_pred = y_pred.round()
        y_pred = y_pred.detach().numpy()
    
    return y_pred

