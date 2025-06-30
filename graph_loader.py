import os
import torch
import time
from typing import Tuple, Optional
from graph_creator import GraphCreator
from args import Args

class GraphLoader:
    """
    Hanterar generering och caching av grafer för PyTorch.
    """
    def __init__(self, args: Args, cache_dir: str = "./graphs"):
        self.args = args
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Skapa en temporär GraphCreator för att få ett unikt filnamn
        self.creator = GraphCreator(self.args)
        self.filename = os.path.splitext(self.creator.filename)[0] + ".pt"
        self.cache_path = os.path.join(self.cache_dir, self.filename)

    def save_batch(self, batch_data: Tuple[torch.Tensor, ...]) -> None:
        """Sparar batch till fil."""
        torch.save(batch_data, self.cache_path)
        print(f"Batch sparad till {self.cache_path}")

    def load_batch(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """Försöker ladda batch från cache."""
        if os.path.isfile(self.cache_path):
            t0 = time.perf_counter()
            batch_data = torch.load(self.cache_path, map_location=self.args.device, weights_only=True)
            print(f"Batch laddad från {self.cache_path} på {time.perf_counter()-t0:.2f} s")
            return batch_data
        return None

    def get_batches(self, force_refresh: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Returnerar batch, antingen från cache eller genom att generera den.

        Args:
            force_refresh: Om True, tvingar ny generering.

        Returns:
            Tuple med PyTorch-tensorer.
        """
        if not force_refresh:
            cached = self.load_batch()
            if cached is not None:
                return cached

        print("Genererar ny batch...")
        batch = self.creator.generate_batch()
        self.save_batch(batch)
        return batch

if __name__ == "__main__":
    # Exempel på användning
    args = Args(
        error_rates=[0.002],
        distance=3,
        t=[21],
        sliding=True,
        dt=2,
        batch_size=8,
        seed=42,
        norm=torch.inf
    )
    
    loader = GraphLoader(args)
    batch = loader.get_batches()
    #print(batch[0])
    print("Batch klar för användning.")
