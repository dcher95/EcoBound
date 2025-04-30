import argparse
import os
import sys

import train
import inference
import plot_map
import testing

def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline: training, inference, and visualization.")
    
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--plot", action="store_true", help="Generate plots from results")
    parser.add_argument("--test", action="store_true", help="Test against ground truth")
    
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

    if args.plot:
        print("Testing...")
        testing.main() 

if __name__ == "__main__":
    main()
