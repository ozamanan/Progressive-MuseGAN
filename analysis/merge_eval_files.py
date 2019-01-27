"""This file provides functions for merging evaluation files in a directory
"""
import os
import joblib
import numpy as np

EXP_NAMES = ['test',]

SRC_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'exp', exp_name, 'eval')
    for exp_name in EXP_NAMES
]
print(SRC_DIRS)

DST_PATHS = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                 'eval_training_progress', exp_name + '.npz')
    for exp_name in EXP_NAMES
]

def get_npy_files(target_dir):
    """Return a list of paths to all the .npy files in a directory."""
    filepaths = []
    for path in os.listdir(target_dir):
        if path.endswith('.npy'):
            filepaths.append(path)
    return filepaths

def load(filepath, eval_dir):
    """Load a evaluation file at the given path and return the stored data."""
    step = int(filepath.split("_")[0])
    print(step)
    print("===========")
    data = np.load(os.path.join(eval_dir, filepath))
    return (step, data[()]['score_matrix_mean'],
            data[()]['score_pair_matrix_mean'])

def main():
    """Main function"""
    for idx, eval_dir in enumerate(SRC_DIRS):
        filepaths = get_npy_files(eval_dir)
        collected = joblib.Parallel(n_jobs=30, verbose=5)(
            joblib.delayed(load)(filepath, eval_dir) for filepath in filepaths)

        steps = []
        score_matrix_means = []
        score_pair_matrix_means = []

        for item in collected:
            steps.append(item[0])
            score_matrix_means.append(item[1])
            score_pair_matrix_means.append(item[2])

        steps = np.array(steps)
        score_matrix_means = np.stack(score_matrix_means)
        score_pair_matrix_means = np.stack(score_pair_matrix_means)

        argsort = steps.argsort()
        steps = steps[argsort]
        score_matrix_means = score_matrix_means[argsort]
        score_pair_matrix_means = score_pair_matrix_means[argsort]

        np.savez(DST_PATHS[idx], steps=steps,
            score_matrix_means=score_matrix_means, score_pair_matrix_means=score_pair_matrix_means)

if __name__ == "__main__":
    main()
