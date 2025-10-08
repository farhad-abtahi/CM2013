#!/usr/bin/env python3
"""
Script to visualize EDF signals and hypnogram from XML annotations.

Usage:
    python plot_signals.py --edf <path_to_edf> --xml <path_to_xml> [--epoch <epoch_number>]

Example:
    python plot_signals.py --edf data/training/file.edf --xml data/training/file.xml --epoch 100
"""

import argparse
from src.visualization import plot_sample_epoch, plot_hypnogram

def main():
    parser = argparse.ArgumentParser(description='Visualize sleep signals and hypnogram')
    parser.add_argument('--edf', type=str, help='Path to EDF file')
    parser.add_argument('--xml', type=str, help='Path to XML annotation file')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number to plot (default: 0)')
    parser.add_argument('--epoch-duration', type=int, default=30, help='Epoch duration in seconds (default: 30)')

    args = parser.parse_args()

    # Plot signals from EDF file
    if args.edf:
        print(f"Plotting signals from EDF file: {args.edf}")
        plot_sample_epoch(args.edf, epoch_idx=args.epoch, epoch_duration=args.epoch_duration)
    else:
        print("No EDF file specified. Use --edf to specify a file.")

    # Plot hypnogram from XML file
    if args.xml:
        print(f"\nPlotting hypnogram from XML file: {args.xml}")
        plot_hypnogram(args.xml, edf_path=args.edf)
    else:
        print("No XML file specified. Use --xml to specify a file.")

    # If neither specified, show usage
    if not args.edf and not args.xml:
        parser.print_help()
        print("\n" + "="*70)
        print("Example usage:")
        print("  python plot_signals.py --edf data/training/recording1.edf --xml data/training/recording1.xml")
        print("  python plot_signals.py --edf data/training/recording1.edf --epoch 100")
        print("  python plot_signals.py --xml data/training/recording1.xml")
        print("="*70)

if __name__ == "__main__":
    main()
