import argparse
import os
import sys

import train
import inference
import plot_map

def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline: training, inference, and visualization.")
    
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--plot", action="store_true", help="Generate plots from results")
    
    args = parser.parse_args()

    if args.train:
        print("Starting training...")
        train.main()  

    if args.infer:
        print("Running inference...")
        inference.main()  

    if args.plot:
        print("Generating plots...")
        plot_map.main()  

if __name__ == "__main__":
    main()
