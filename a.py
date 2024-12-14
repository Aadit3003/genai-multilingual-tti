import argparse
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quox", type=str, help='Path where Trained RKS-Diffusion model checkpoints are stored')
    args = parser.parse_args()
    quox = args.quox
    print("QUOX ", quox)
    
if __name__ == '__main__':
    main()