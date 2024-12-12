from mpi4py import MPI
import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt

def compute_histogram(image_chunk):
    """Обчислення гістограми для заданого блоку зображення."""
    histogram, _ = np.histogram(image_chunk, bins=256, range=(0, 256))
    return histogram

def main():
    #-------------------------------------------------------Ініціалізація MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if len(sys.argv) < 2:
            print("Usage: python histogram_mpi.py <image_path>")
            sys.exit(1)

        image_path = sys.argv[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # ------------------------Завантаження зображення в градаціях сірого
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            sys.exit(1)

        height, width = image.shape
        chunk_size = height // size
        extra_rows = height % size

        #-----------------------------------------------------------Поділ зображення на частини
        chunks = [image[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        if extra_rows > 0:
            chunks[-1] = np.vstack((chunks[-1], image[-extra_rows:]))
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)

    local_histogram = compute_histogram(chunk)

    # -----------------------------Об'єднання локальних гістограм у загальну -------------------------
    total_histogram = np.zeros(256, dtype=int)
    comm.Reduce(local_histogram, total_histogram, op=MPI.SUM, root=0)

    # Ранг 0 зберігає результати та виводить час виконання
    comm.Barrier()
    if rank == 0:
        output_csv = "histogram.csv"
        with open(output_csv, "w") as f:
            f.write("Brightness,Count\n")
            for i, count in enumerate(total_histogram):
                f.write(f"{i},{count}\n")
        print(f"Histogram saved to {output_csv}")

        # Візуалізація гістограми
        output_png = 'histogram.png'
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), total_histogram, color='gray')
        plt.title("Image Brightness Histogram")
        plt.xlabel("Pixel Brightness")
        plt.ylabel("Pixel Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(output_png)
        print(f"Histogram image saved to {output_png}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    # Загальний час виконання програми
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Total execution time: {end_time - start_time:.4f} seconds")
