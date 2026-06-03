# ============================================================
# CLUSTERSENSE AI — MAIN ENTRY POINT
# Run: python main.py
# ============================================================

from src.clustersense_package.pipeline import run_pipeline


def main():

    print("Starting ClusterSense AI Pipeline...")

    best_k, best_score = run_pipeline()

    print(f"\nBest Clusters         : {best_k}")
    print(f"Best Silhouette Score : {best_score:.4f}")
    print("\nPipeline Execution Completed Successfully.")


if __name__ == "__main__":

    main()