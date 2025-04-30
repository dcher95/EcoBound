import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score


from config import config

#--------------------------------   
# SEARCH LOOP
#--------------------------------


def search_loop_multi(
    species_list: list,
    method: str = "random",
    verbose: bool = True,
    sampds: pd.DataFrame = None,
    extrastr: str = "",
    lambda_val = 0
):
    """
    Active‐sampling loop over multiple species at once.
    Sampling for 'uncertainty' or 'costaware' uses the *average*
    uncertainty across species_list.
    Records final simplex‐model weights, mAP, and AUC per species.
    """
    # -- Initialize globals if needed --
    global all_map_scores, all_auc_scores, all_weights
    if 'all_map_scores' not in globals():
        all_map_scores = {}
    if 'all_auc_scores' not in globals():
        all_auc_scores = {}
    if 'all_weights' not in globals():
        all_weights = {}
    trip_stops = {}

    # Prepare nested dicts for each species & method
    method_key = f"{method}{extrastr}"
    for sp in species_list:
        all_map_scores.setdefault(sp, {}).setdefault(method_key, [])
        all_auc_scores.setdefault(sp, {}).setdefault(method_key, [])
        all_weights.setdefault(sp, {})[method_key] = None

    # 1) initialize posterior & uncertainty columns for each species
    for sp in species_list:
        pcol = f"posterior_{method}_{sp}{extrastr}"
        ucol = f"uncertainty_{method}_{sp}{extrastr}"
        sampds[pcol] = sampds[f"prob_{sp}{extrastr}"]
        sampds[ucol] = 0.5 - (0.5 - sampds[pcol]).abs()

    # 2) compute avg‐uncertainty column
    ucols = [f"uncertainty_{method}_{sp}{extrastr}" for sp in species_list]
    sampds[f"avg_uncertainty_{method}{extrastr}"] = sampds[ucols].fillna(0).mean(axis=1) 

    
    # keep track of where we sample
    sampled_indices = []
    for trip in range(config.avail_trips):
        for sp in species_list:
            trip_stops.setdefault(sp, {})
            trip_stops[sp][trip] = []
                
        if verbose:
            print(f"\n--- Trip {trip+1}/{config.avail_trips} ({method}) ---")

        # reset trip
        cur_lat, cur_lon = config.start_lat, config.start_lon
        remaining_seconds = config.seconds_per_trip

        # update distance & sample_time
        sampds['distance'] = np.sqrt((sampds['lon']-cur_lon)**2 + (sampds['lat']-cur_lat)**2)
        sampds['sample_time'] = sampds['distance']*config.seconds_per_degree + config.seconds_to_sample

        # restrict to reachable & unsampled
        reachable = sampds[
            (sampds['sample_time'] <= remaining_seconds) &
            (~sampds.index.isin(sampled_indices))
        ].copy()
        if reachable.empty:
            if verbose: print(" No reachable points, skipping trip.")
            continue

        # inner sampling loop
        while remaining_seconds > 0 and not reachable.empty:
            if method == "random":
                pick = reachable.sample(n=1, random_state=config.seed).iloc[0]
            else:
                # use avg‐uncertainty for ranking
                if method == "uncertainty":
                    score = reachable[f"avg_uncertainty_{method}{extrastr}"]
                else:  # costaware
                    score = reachable[f"avg_uncertainty_{method}{extrastr}"] \
                          - lambda_val * reachable['distance']
                pick = reachable.loc[[score.idxmax()]].iloc[0]

            idx = pick.name
            sampled_indices.append(idx)

            # move there
            cur_lat, cur_lon = pick['lat'], pick['lon']
            remaining_seconds -= pick['sample_time']

            for sp in species_list:
                trip_stops[sp][trip].append({
                    "index": idx,
                    "lat": cur_lat,
                    "lon": cur_lon
                })

            # update posterior for each species at point idx using eBird data
            for sp in species_list:
                sampds.loc[idx, f"posterior_{method}_{sp}{extrastr}"] = pick[sp]

            # recompute reachable
            reachable = reachable.drop(idx)
            reachable['distance'] = np.sqrt((reachable['lon']-cur_lon)**2 + (reachable['lat']-cur_lat)**2)
            reachable['sample_time'] = reachable['distance']*config.seconds_per_degree + config.seconds_to_sample
            reachable = reachable[reachable['sample_time'] <= remaining_seconds]

        # --- end inner loop: now fit models & update posteriors/uncertainties ---
        sampled_df = sampds.loc[sampled_indices]
        prior_cols = [c for c in sampds.columns if c.startswith('prob_')]

        # for each species, fit simplex‐model, record metrics
        for sp in species_list:
            # prepare X & y
            X = sampled_df[prior_cols]
            y = sampled_df[sp].dropna()
            X = X.loc[y.index]

            # drop any NaNs
            if X.isna().any().any():
                clean = X.dropna().index
                X = X.loc[clean]
                y = y.loc[clean]

            # objective & constraints
            n_feat = X.shape[1]
            x0 = np.ones(n_feat)/n_feat
            cons = [{'type':'eq','fun': lambda w: w.sum()-1}]
            bnds = [(0,1)]*n_feat

            # seperate f(n)
            def obj(w): return np.sum((X @ w - y)**2)

            res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons)
            if not res.success:
                raise RuntimeError(f"opt failed for {sp}: {res.message}")
            w = res.x

            # save final weights (will overwrite until last trip, so final remains)
            all_weights[sp][method_key] = w

            # predict on all
            X_all = sampds[prior_cols]
            p_all = np.clip(X_all @ w, 0, 1)
            sampds[f"posterior_{method}_{sp}{extrastr}"] = p_all
            sampds[f"uncertainty_{method}_{sp}{extrastr}"] = 0.5 - (0.5 - p_all).abs()

            # metrics
            mask = ~sampds[sp].isna()
            mAP = average_precision_score(sampds.loc[mask, sp], p_all[mask])
            try:
                AUC = roc_auc_score(sampds.loc[mask, sp], p_all[mask])
            except ValueError:
                AUC = np.nan

            all_map_scores[sp][method_key].append(mAP)
            all_auc_scores[sp][method_key].append(AUC)

        # recompute avg‐uncertainty for next trip
        sampds[f"avg_uncertainty_{method}{extrastr}"] = sampds[ucols].mean(axis=1)

    # done all trips
    return all_weights, all_map_scores, all_auc_scores, trip_stops
