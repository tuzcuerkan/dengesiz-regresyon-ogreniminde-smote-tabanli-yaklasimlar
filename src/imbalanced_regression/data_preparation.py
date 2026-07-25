from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DEFAULT = Path(os.environ.get('THESIS_RAW_DATA_DIR', REPOSITORY_ROOT / 'data' / 'raw'))
SPECIAL = (-88, -90, -99)
AQ_MAIN = ['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
AQ_EXTENDED = AQ_MAIN + ['CO(GT)','C6H6(GT)','NOx(GT)']
BMI_CONTINUOUS = ['age']
BMI_ORDINAL = ['educlt','income','incomehh','health']
BMI_NOMINAL = ['gender','marital','nuts1']
BMI_BINARY = ['empstat1','empstat2','empstat3','empstat4','empstat5','empstat6','empstat7','empstat8','empstatot']
INCOME_TO_TL = {1:0.,2:2500.,3:6250.,4:8750.,5:11250.,6:13750.,7:16000.,8:18500.,9:22500.,10:27500.,11:32500.,12:37500.,13:45000.,14:55000.,15:65000.,16:75000.,17:85000.,18:95000.,19:112500.,20:137500.,21:162500.,22:187500.,23:212500.}


@dataclass(frozen=True)
class DatasetBundle:
    key: str
    display_name: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    continuous: tuple[str, ...] = ()
    ordinal: tuple[str, ...] = ()
    nominal: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()
    fixed_categories: dict[str, tuple[Any, ...]] | None = None
    notes: str = ''

    def summary(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'display_name': self.display_name,
            'row_count': int(len(self.y)),
            'raw_feature_count': int(self.X.shape[1]),
            'group_count': int(self.groups.nunique()),
            'duplicate_predictor_extras': int(len(self.groups) - self.groups.nunique()),
            'feature_missing_cells': int(self.X.isna().sum().sum()),
            'continuous': list(self.continuous),
            'ordinal': list(self.ordinal),
            'nominal': list(self.nominal),
            'binary': list(self.binary),
            'target_min': float(self.y.min()),
            'target_max': float(self.y.max()),
            'notes': self.notes,
        }


def stable_groups(X: pd.DataFrame) -> pd.Series:
    safe = X.copy().where(pd.notna(X), '<NA>').astype(str)
    text = safe.agg('\x1f'.join, axis=1)
    out = text.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:20])
    out.name = 'duplicate_group'
    return out


def _numeric(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    v = pd.to_numeric(s, errors='coerce').replace(list(SPECIAL), np.nan)
    if lo is not None and hi is not None:
        v = v.where(v.between(lo, hi))
    return v.astype(float)


def _yes_no(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').map({1:1.0,2:0.0})


def load_abalone(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df = pd.read_csv(raw/'abalone/abalone.data', header=None, names=['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','Rings'])
    X = pd.get_dummies(df.drop(columns='Rings'), columns=['sex'], dtype=float)
    y = df['Rings'].astype(float).rename('Rings')
    return DatasetBundle('abalone','Abalone',X,y,stable_groups(X),continuous=tuple(c for c in X if not c.startswith('sex_')),binary=tuple(c for c in X if c.startswith('sex_')),notes='Fixed F/I/M one-hot encoding.')


def load_california(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df = pd.read_csv(raw/'california_housing/california_housing.csv')
    target = 'MedHouseVal' if 'MedHouseVal' in df else df.columns[-1]
    X = df.drop(columns=target).astype(float)
    y = df[target].astype(float).rename(target)
    return DatasetBundle('california_housing','California Housing',X,y,stable_groups(X),continuous=tuple(X.columns))


def load_concrete(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df = pd.read_excel(raw/'concrete/Concrete_Data.xlsx')
    X = df.iloc[:,:-1].copy(); X.columns = X.columns.astype(str)
    y = pd.to_numeric(df.iloc[:,-1],errors='raise').astype(float).rename(str(df.columns[-1]))
    return DatasetBundle('concrete','Concrete Compressive Strength',X,y,stable_groups(X),continuous=tuple(X.columns))


def load_wine(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df = pd.read_csv(raw/'wine_quality/winequality-red.csv', sep=';')
    X = df.drop(columns='quality').astype(float); y=df['quality'].astype(float).rename('quality')
    return DatasetBundle('wine_quality_red','Wine Quality Red',X,y,stable_groups(X),continuous=tuple(X.columns))


def load_air_quality(raw: Path = RAW_DEFAULT, scope: str = 'main') -> DatasetBundle:
    df = pd.read_excel(raw/'air_quality_no2/AirQualityUCI.xlsx').replace(-200,np.nan)
    df = df.loc[df['NO2(GT)'].notna()].reset_index(drop=True)
    cols = AQ_MAIN if scope == 'main' else AQ_EXTENDED
    X = df[cols].apply(pd.to_numeric,errors='coerce')
    y = pd.to_numeric(df['NO2(GT)'],errors='raise').astype(float).rename('NO2(GT)')
    key = 'air_quality_no2' if scope == 'main' else 'air_quality_no2_extended'
    name = 'Air Quality NO2' if scope == 'main' else 'Air Quality NO2 extended sensitivity'
    return DatasetBundle(key,name,X,y,stable_groups(X),continuous=tuple(cols),notes=f'{scope} scope; target-valid rows retained; -200 as missing.')


def load_servo(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df = pd.read_csv(raw/'servo/servo.data',header=None,names=['motor','screw','pgain','vgain','class'])
    X = pd.get_dummies(df.drop(columns='class'),columns=['motor','screw'],dtype=float)
    y = df['class'].astype(float).rename('class')
    return DatasetBundle('servo','Servo',X,y,stable_groups(X),ordinal=('pgain','vgain'),binary=tuple(c for c in X if c.startswith(('motor_','screw_'))),notes='Fixed A-E one-hot encoding.')


def load_tgss(raw: Path = RAW_DEFAULT) -> pd.DataFrame:
    return pd.read_csv(raw/'tgss/TGSS2024.csv',low_memory=False)


def load_bmi(raw: Path = RAW_DEFAULT, schema: str = 'primary') -> DatasetBundle:
    df = load_tgss(raw)
    yall = pd.to_numeric(df['bmi'],errors='coerce')
    df = df.loc[yall.between(10,60)].copy().reset_index(drop=True)
    y = pd.to_numeric(df['bmi'],errors='raise').astype(float).rename('bmi')
    if schema == 'workstat_sensitivity':
        cols=['age','gender','educlt','income','incomehh','marital','nuts1','health','workstat']
        X=df[cols].apply(pd.to_numeric,errors='coerce').replace(list(SPECIAL),np.nan)
        return DatasetBundle('tgss_bmi_workstat_sensitivity','TGSS BMI work-status sensitivity',X,y,stable_groups(X),continuous=('age',),ordinal=('educlt','income','incomehh','health'),nominal=('gender','marital','nuts1','workstat'),fixed_categories={'gender':(1,2),'marital':tuple(range(1,7)),'nuts1':tuple(range(1,13)),'workstat':tuple(range(1,10))},notes='Sensitivity specification; workstat is structurally missing for most rows.')
    if schema != 'primary':
        raise ValueError(f'Bilinmeyen TGSS BMI şeması: {schema}')
    X=pd.DataFrame(index=df.index)
    X['age']=_numeric(df['age'],18,120)
    X['educlt']=_numeric(df['educlt'],1,9)
    X['income']=_numeric(df['income'],1,23)
    X['incomehh']=_numeric(df['incomehh'],1,24)
    X['health']=_numeric(df['health'],1,5)
    X['gender']=_numeric(df['gender'],1,2)
    X['marital']=_numeric(df['marital'],1,6)
    X['nuts1']=_numeric(df['nuts1'],1,12)
    for c in BMI_BINARY:
        X[c]=_yes_no(df[c])
    assert X[BMI_BINARY].isna().sum().sum()==0
    return DatasetBundle('tgss_bmi','TGSS BMI',X,y,stable_groups(X),continuous=tuple(BMI_CONTINUOUS),ordinal=tuple(BMI_ORDINAL),nominal=tuple(BMI_NOMINAL),binary=tuple(BMI_BINARY),fixed_categories={'gender':(1,2),'marital':tuple(range(1,7)),'nuts1':tuple(range(1,13))},notes='TGSS BMI schema; workstat removed, complete employment indicators retained.')


def load_income(raw: Path = RAW_DEFAULT) -> DatasetBundle:
    df=load_tgss(raw)
    yall=pd.to_numeric(df['income'],errors='coerce').map(INCOME_TO_TL)
    valid=yall.notna(); d=df.loc[valid].copy(); y=yall.loc[valid].astype(float).reset_index(drop=True).rename('approx_personal_monthly_net_income_tl')
    X=pd.DataFrame(index=d.index)
    X['age']=_numeric(d['age'],18,120)
    X['gender_female']=pd.to_numeric(d['gender'],errors='coerce').map({1:0.,2:1.})
    X['education_level']=_numeric(d['educlt'],1,9)
    X['household_members_excl_self']=_numeric(d['hhsize'],0,15)
    X['self_rated_health']=_numeric(d['health'],1,5)
    X['urbanisation_level']=_numeric(d['degurba'],1,3)
    X['health_insurance']=_yes_no(d['sscover'])
    e1=_yes_no(d['empstat1']); ec=_yes_no(d['empchek'])
    X['currently_working']=np.select([e1.eq(1)|ec.eq(1),e1.eq(0)&(ec.eq(0)|ec.isna())],[1.,0.],default=np.nan)
    mapping={'student':'empstat2','unemployed_active':'empstat3','unemployed_inactive':'empstat4','permanent_illness_disability':'empstat5','retired':'empstat6','compulsory_military':'empstat7','home_care_work':'empstat8','other_employment_status':'empstatot'}
    for out,src in mapping.items(): X[out]=_yes_no(d[src])
    wh=_numeric(d['workhrsw'],1,120); X['weekly_work_hours']=np.where(X['currently_working'].eq(0),0.,wh)
    X=X.reset_index(drop=True)
    return DatasetBundle('tgss_income','TGSS approximate personal monthly net income',X,y,stable_groups(X),continuous=('age','household_members_excl_self','weekly_work_hours'),ordinal=('education_level','self_rated_health','urbanisation_level'),binary=tuple(c for c in X if c not in {'age','household_members_excl_self','weekly_work_hours','education_level','self_rated_health','urbanisation_level'}),notes='23 income categories mapped to representative TL values.')


def load_all_main(raw: Path = RAW_DEFAULT) -> dict[str, DatasetBundle]:
    bundles=[load_abalone(raw),load_california(raw),load_concrete(raw),load_wine(raw),load_air_quality(raw,'main'),load_servo(raw),load_bmi(raw,'primary'),load_income(raw)]
    return {b.key:b for b in bundles}


def target_bins(y: pd.Series, n_splits: int) -> np.ndarray:
    q=min(max(n_splits,5),10)
    try:
        b=pd.qcut(y.rank(method='first'),q=q,labels=False,duplicates='drop')
    except ValueError:
        b=pd.cut(y,bins=q,labels=False,duplicates='drop')
    arr=np.asarray(b,dtype=int)
    if np.bincount(arr).min() < n_splits:
        q=max(2, min(5, len(y)//n_splits))
        arr=np.asarray(pd.qcut(y.rank(method='first'),q=q,labels=False,duplicates='drop'),dtype=int)
    return arr


def make_grouped_splits(bundle: DatasetBundle, n_splits: int, seeds: tuple[int,...]) -> list[dict[str,Any]]:
    # Split at the unique-group level, stratifying groups by median target quantiles.
    frame=pd.DataFrame({'group':bundle.groups.astype(str),'y':bundle.y.to_numpy(),'row':np.arange(len(bundle.y))})
    group_table=frame.groupby('group',sort=True).agg(group_y=('y','median'),group_n=('row','size')).reset_index()
    bins=target_bins(group_table['group_y'],n_splits)
    results=[]
    for repeat,seed in enumerate(seeds,1):
        cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
        for fold,(gtrain,gtest) in enumerate(cv.split(group_table,bins),1):
            train_group_set=set(group_table.iloc[gtrain]['group'])
            test_group_set=set(group_table.iloc[gtest]['group'])
            train=np.flatnonzero(frame['group'].isin(train_group_set).to_numpy())
            test=np.flatnonzero(frame['group'].isin(test_group_set).to_numpy())
            overlap=train_group_set.intersection(test_group_set)
            if overlap: raise RuntimeError(f'{bundle.key}: group overlap')
            test_y=bundle.y.iloc[test]
            rare_cut=float(bundle.y.quantile(0.9))
            results.append({'repeat':repeat,'fold':fold,'seed':seed,'train_index':train,'test_index':test,'train_n':len(train),'test_n':len(test),'train_group_n':len(train_group_set),'test_group_n':len(test_group_set),'group_overlap_n':0,'test_upper_decile_n':int((test_y>=rare_cut).sum())})
    return results
