"""Right-censoring check + more permutations for a real p-value.

With a 52-week label and a panel ending 2026-07-27, any row after ~2025-07-27
cannot have its full outcome window observed. Those rows are labelled 0 even
when a deal is coming, which DEPRESSES measured precision. The folds (52-week
test windows, mean 35.7%) far exceeded the full-period test (22.3%), and this
is the most likely reason.
"""
import datetime as dt, gc, json
import numpy as np, polars as pl
from deal import features, model_gbm, screen
import sys; sys.path.insert(0, 'scripts')
from final_stats import relabel, HORIZON, N_EVAL, TEST_START

raw = pl.read_parquet('data/features.parquet')
usable=[c for c in features.FEATURE_COLS if raw[c].std() and raw[c].std()>0]
cols=[c for c in usable if not c.endswith('_z')]
df = relabel(raw, HORIZON); del raw; gc.collect()

PANEL_END = df['week'].max()
SAFE_END = PANEL_END - dt.timedelta(weeks=HORIZON)
print(f"panel ends {PANEL_END} | fully-observed outcomes only through {SAFE_END}\n")

tr = df.filter(pl.col('week') < TEST_START)
te_all = df.filter(pl.col('week') >= TEST_START)
te_safe = df.filter((pl.col('week') >= TEST_START) & (pl.col('week') <= SAFE_END))

def sc(tr_, te_, cs=cols):
    b = model_gbm.fit(tr_, valid=te_, cols=cs)
    r = screen.weekly_precision(te_, model_gbm.predict(b, te_, cs), N_EVAL)
    del b; gc.collect(); return r['precision'], r['lift'], r['base_rate']

pa, la, ba = sc(tr, te_all)
ps, ls, bs_ = sc(tr, te_safe)
print(f"test ALL   ({te_all.height:,} rows, base {ba*100:.2f}%): prec@25 {pa*100:.2f}%  lift {la:.2f}x")
print(f"test SAFE  ({te_safe.height:,} rows, base {bs_*100:.2f}%): prec@25 {ps*100:.2f}%  lift {ls:.2f}x")
print(f"censoring cost: {(ps-pa)*100:+.2f}pp\n")

print("=== permutation test, 20 draws (uncensored test set) ===", flush=True)
null=[]
for k in range(20):
    sh = tr.with_columns(pl.col('y').shuffle(seed=2000+k).over('week').alias('y'))
    p,_,_ = sc(sh, te_safe); null.append(p)
    print(f"  null {k+1:>2}: {p*100:.2f}%", flush=True)
    del sh; gc.collect()
null=np.array(null)
p_val = (np.sum(null >= ps) + 1)/(len(null)+1)
print(f"\n  real {ps*100:.2f}%  |  null mean {null.mean()*100:.2f}%  max {null.max()*100:.2f}%  sd {null.std()*100:.2f}")
print(f"  p = {p_val:.4f}   beats all null draws: {bool(ps>null.max())}")
json.dump({'prec_all':float(pa),'prec_safe':float(ps),'lift_safe':float(ls),
           'base_safe':float(bs_),'null_mean':float(null.mean()),
           'null_max':float(null.max()),'p_value':float(p_val)},
          open('data/censor_check.json','w'), indent=2)
