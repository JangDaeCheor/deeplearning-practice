# =====================================================================
#  04. 사이킷런 — 01~03 에서 손으로 한 일을 '한 줄'로, 그리고 언제 쓰나
#  실행: python 04_사이킷런_같은일을_한줄로.py   (수업코드 폴더에서)
# =====================================================================
# 사이킷런(scikit-learn, 코드에선 sklearn) = 파이썬 머신러닝 도구 상자.
# 우리가 01~03에서 손으로 짠 것(표준화 경사하강 회귀 분류 채점)이 전부 함수로 들어 있습니다.
# 왜 손으로 먼저 했나? 도구가 '안에서 무슨 일을 하는지' 알아야 결과를 읽고, 이상할 때 고칠 수 있어서.
# 언제 쓰나? 표(엑셀 같은)데이터 + 선형모델 트리 같은 전통 머신러닝. 실무 첫 선택은 거의 이것.
# "손으로 한 값 == 사이킷런 값"을 눈으로 확인하고, 규칙 3개를 몸에 익힌다.
# 규칙 1) X는 (설비 수, 센서 수) 표 모양, y는 한줄
# 규칙 2) 만들기 -> fit(학습) -> predict(예측) 세 동사
# 규칙 3) 학습 결과는 이름 끝에 밑줄: coef_, intercept_

import os
import numpy as np
import pandas as pd

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")
특징이름 = ["공기온도", "회전수", "토크", "공구마모"]


# =====================================================================
# 0. 사이킷런에서 꺼내 쓰는 것들 — 미리 한눈에
# =====================================================================
# 1. 회귀 — 01 의 "공기온도 → 공정온도" 를 세 줄로
# =====================================================================
from sklearn.linear_model import LinearRegression

x = df["공기온도"].values
y = df["공정온도"].values
X1 = x.reshape(-1, 1)
# 규칙 1: 센서가 하나여도 (200, 1) '세로 표'로 세운다.
# reshape(-1, 1)의 -1 = "행 수는 알아서 맞춰"
print("[1] x.shape", x.shape, "→ X1.shape", X1.shape)
# 왜? 사이킷런은 "행 = 설비, 열 = 센서" 표만 받습니다. 센서 1개면 열이 1개인 표. 한 줄짜리 배열은 안 받음.

# 규칙 2 : 만들고 학습
model = LinearRegression()  # ① 만들기 — 빈 선형회귀 모델. 아직 아무것도 모름
model.fit(X1, y)  # ② 학습 — 01 에서 300 걸음 걸은 그 일이 이 한 줄

# 규칙 3 : 학습에서 얻은 값은 끝에 밑줄(_). coef_ = w 들, intercept_ = b
print(
    f"    w = {model.coef_[0]:.4f}  b = {model.intercept_:.4f}"
)  # 규칙 3: 학습해서 얻은 값은 끝에 밑줄(_). coef_ = w 들, intercept_ = b
print("    01 에서 걸어서 찾은 값: w = 0.9840, b = 14.7799  ← 같죠?")
print(
    "    공기온도 300 → 예측:", round(model.predict([[300.0]])[0], 2)
)  # ③ 예측. 새 데이터도 '표 모양'이라 대괄호 두 겹 [[ ]]
# .score = 회귀면 R², 분류면 정확도
# 사이킷런 LinearRegression은 걷지 않고 '공식'으로 단번에 풉니다. 그래서 lr epoch가 없음.
# 선형회귀(와 그 사촌 Ridge)는 공식이 있고, 로지스틱 회귀부터는 사이킷런도 속으로 걷습니다.
# (max_iter 가 그 흔적)
print("    R²:", round(model.score(X1, y), 4), "(01 의 0.7989)")
# 가장 흔한 에러 미리 보기 X를 한 줄(1차원)로 넣으면:

try:
    LinearRegression().fit(x, y)
except ValueError as e:
    print("    [에러] ", str(e).splitlines()[0], "→ reshape(-1, 1) 하라는 뜻")


# =====================================================================
# 2. 나누기 + 표준화 + 다변수 회귀 — 02 를 도구로
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[특징이름].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 02의 "섞고 70:30으로 자르기"가 이 한 줄. 돌려주는 순서 (X학습, X시험, y학습, y시험)을 외우세요.
# 순서 바꾸면 조용히 망합니다.
# random_state=42 = 02의 RandomState(42)와 같은 역할(재현성).
print("\n[2] train", X_train.shape, "/ test", X_test.shape)

scaler = StandardScaler()  # 표준화 도구
scaler.fit(X_train)  # 학습용의 평균·표준편차를 '외운다'  (02 의 mu, sd 계산)
Z_train = scaler.transform(X_train)  # 외운 값으로 변환  (02 의 (X − mu) / sd)
Z_test = scaler.transform(X_test)  # 시험용도 학습용 눈금으로 '변환만'
# *(scaler.fit(X_test) 하면 누수!)*
print(
    "    scaler 가 외운 평균:",
    scaler.mean_.round(1),
    "← 02 의 mu 역할 (분할이 달라 값은 조금 다름)",
)

reg = LinearRegression().fit(
    Z_train, y_train
)  # 만들기와 fit 을 한 줄에 붙여 쓰기도 합니다
print("    표준화 가중치:", dict(zip(특징이름, reg.coef_.round(3).tolist())))
# dict(zip(이름, 값)) = 이름:값 짝 사전
# 02와 분할이 달라 숫자는 조금 다르지만 그림은 같습니다
print(
    f"    train R² {reg.score(Z_train, y_train):.4f} / test R² {reg.score(Z_test, y_test):.4f}"
)
# train 0.816 / test 0.743 -> 차이 0.073. 02의 경보선(0.005 정상 / 0.10 의심) 사이 = '경계'
# 시험용이 60대뿐이라 흔들리는 정도.
# 진짜 과적합인지 확인하려면 교차검증에서 다시 봅니다 밑에서.


# =====================================================================
# 3. Pipeline — "표준화는 학습용으로만" 을 도구가 대신 지키게
# =====================================================================
from sklearn.pipeline import make_pipeline

# 파이프라인 = 전처리(표준화 등)와 모델을 순서대로 이어 붙여 '하나의 모델'처럼 다루는 것.
# 왜? fit 할 땐 학습용으로 스케일러를 맞추고, predict 할 땐 변환만 하는 걸 도구가 알아서 해줌.
# 사람이 실수로 scaler.fit(X_test)를 못하게 구조로 막는 것.
# 그리고 새 데이터도 '원래 눈금'그대로 넣으면 됩니다.
pipe = make_pipeline(
    StandardScaler(), LinearRegression()
)  # 왼쪽부터 순서대로: 표준화 → 선형회귀
pipe.fit(X_train, y_train)  # 원래 눈금 X 를 그냥 넣는다. 안에서 표준화까지 함
print("\n[3] Pipeline test R²:", round(pipe.score(X_test, y_test), 4), "(위와 같음)")
print(
    "    새 설비 [공기 300, 회전 1500, 토크 40, 마모 100] → 공정온도",
    round(pipe.predict([[300, 1500, 40, 100]])[0], 2),
)
# 함정 누수(leakage)가 얼마나 속이는지 눈으로 봄
# 누수 = 시험용 데이터의 정보가 학습 쪽으로 새어 들어가는 것. 점수가 부풀려지고, 현장에서 무너짐.
# 제일 흔한 사고 : 00에서 배운 '중복 행 제거'를 깜빡한 경우.
# 같은 측정이 두 번 들어 있으면, 무작위로 나눌 때 쌍둥이가 학습용과 시험용에 갈라져 들어갑니다.
# 그럼 모델은 시험 문제를 이미 학습용에서 본 셈이 됩니다.

from sklearn.neighbors import KNeighborsRegressor
# KNeighborsRegressor(1) = "제일 비슷한 설비 한 대를 찾아 그 설비의 정답을 그대로 말하는" 단순
# 외우기에 특화돼 있어서 누수를 드러내기 좋습니다.

이웃 = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))

X중복 = np.vstack([X, X])  # 일부러 모든 행을 두 번씩 (중복 제거를 깜빡한 상황)
y중복 = np.concatenate([y, y])
d_tr, d_te, dy_tr, dy_te = train_test_split(
    X중복, y중복, test_size=0.3, random_state=42
)
print("\n[3-1] 중복 행을 안 지우고 나누면 (누수)")
print(
    f"    중복 있음 test R² {이웃().fit(d_tr, dy_tr).score(d_te, dy_te):.4f}   <- 훌륭해 보인다"
)
print(
    f"    중복 없음 test R² {이웃().fit(X_train, y_train).score(X_test, y_test):.4f}   <- 이게 이 모델의 진짜 실력"
)
print(
    "    같은 모델, 같은 데이터입니다. 중복을 안 지운 것 하나로 점수가 네 배가 됐어요."
)
# 현장에서 중복 제거를 하지 않고 학습 시험 봤을 때
# "개발할 땐 0.9 나왔는데 현장에 넣으니 0.4"라는 대형 사고,
# 원인의 절반이 이겁니다.


# =====================================================================
# 4. 분류 — 03 의 로지스틱 회귀를 도구로
# =====================================================================
from sklearn.linear_model import LogisticRegression

yc = df["고장여부"].values  # 분류 정답: 0/1
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, yc, test_size=0.3, random_state=3, stratify=yc
)  # stratify=yc = 고장 비율을 양쪽에 똑같이. 03 2번에서 손으로 한 일이 단어 하나
print(
    "\n[4] 분류 — 학습용 고장", yc_train.sum(), "대 / 시험용 고장", yc_test.sum(), "대"
)

clf = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000)
)  # max_iter = 최대 걸음 수. 기본 100 이 모자라면 경고가 떠서 1000 으로
clf.fit(Xc_train, yc_train)  # 03 의 sigmoid + 로그손실 + 경사하강 2000 바퀴가 이 한 줄

p = clf.predict_proba(Xc_test)[:, 1]
판정 = clf.predict(
    Xc_test
)  # 0.5 기준 판정. 임계값을 바꾸려면 p 를 직접 자릅니다: (p >= 0.2)
print(
    "    시험용 고장 확률 상위 5:", np.sort(p)[::-1][:5].round(3)
)  # 정렬 → [::-1] 뒤집어 큰 순 (00 미리보기 ⑥) → 앞 5개
print(
    "    정확도:",
    round(clf.score(Xc_test, yc_test), 3),
    "← 03 에서 배웠듯 이것만 보면 속는다",
)


# =====================================================================
# 5. 채점 도구 — 03 에서 손으로 센 네 칸·재현율을 함수로
# =====================================================================
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
)

print(
    "\n[5] 혼동행렬 (사이킷런은 [[정상→정상, 정상→고장], [고장→정상, 고장→고장]] 순서)"
)
print(
    confusion_matrix(yc_test, 판정)
)  # 인자 순서: (실제, 판정). 바꾸면 표가 뒤집힙니다
tn, fp, fn, tp = confusion_matrix(yc_test, 판정).ravel()
print(f"    잡음(TP) {tp}  놓침(FN) {fn}  헛경보(FP) {fp}  통과(TN) {tn}")

print("\n    classification_report — 정밀도·재현율을 한 표에 (고장 행을 보세요)")
print(
    classification_report(yc_test, 판정, target_names=["정상", "고장"], zero_division=0)
)

print("    임계값을 내리면 (03 8번을 도구로)")
for th in [0.5, 0.3, 0.2, 0.1]:
    판정_th = (p >= th).astype(int)
    print(
        f"      임계값 {th}: 재현율 {recall_score(yc_test, 판정_th, zero_division=0):.2f}"
        f"  정밀도 {precision_score(yc_test, 판정_th, zero_division=0):.2f}"
    )
clf_b = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
)
clf_b.fit(Xc_train, yc_train)
판정_b = clf_b.predict(Xc_test)
print(
    f"    class_weight='balanced' 로 다시 학습 → 임계값 0.5 에서 재현율 {recall_score(yc_test, 판정_b):.2f}"
    f"  정밀도 {precision_score(yc_test, 판정_b, zero_division=0):.2f}  (놓침 대신 헛경보를 택한 것)"
)


# =====================================================================
# 6. 규제 — 과적합 처방을 옵션 하나로 (Ridge)
# =====================================================================
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

print("\n[6] Ridge alpha 별 점수 — 이 데이터(센서 4개)에선")
for a in [0.01, 1, 10, 100]:
    r = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X_train, y_train)
    print(
        f"    alpha={a:<5} → train {r.score(X_train, y_train):.4f}  test {r.score(X_test, y_test):.4f}"
    )

print(
    "\n[6-1] 특징을 70개로 부풀리면 (PolynomialFeatures) — 과적합이 생기고, Ridge 가 고친다"
)
for 이름, 모델 in [
    ("규제 없음(LinearRegression)", LinearRegression()),
    ("Ridge(alpha=10)", Ridge(alpha=10)),
]:
    poly = make_pipeline(
        StandardScaler(), PolynomialFeatures(4), StandardScaler(), 모델
    ).fit(X_train, y_train)
    print(
        f"    {이름:26s} → train {poly.score(X_train, y_train):.3f}  test {poly.score(X_test, y_test):.3f}"
    )


# =====================================================================
# 7. 손잡이(alpha 등) 고르기 — test 를 훔쳐보지 않고: 교차검증
# =====================================================================
from sklearn.model_selection import cross_val_score, GridSearchCV

cv = cross_val_score(
    make_pipeline(StandardScaler(), LinearRegression()), X_train, y_train, cv=5
)
print("\n[7] 교차검증 5조각 R²:", cv.round(3), "→ 평균", round(cv.mean(), 4))

grid = GridSearchCV(
    make_pipeline(StandardScaler(), Ridge()),
    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=5,
)
grid.fit(X_train, y_train)  # 후보 5개 × 5조각 = 25번 학습
print(
    "    교차검증으로 고른 alpha:",
    grid.best_params_,
    "/ CV 평균 R²:",
    round(grid.best_score_, 4),
)
print("    그 모델의 test R² (이제 딱 한 번):", round(grid.score(X_test, y_test), 4))


# =====================================================================
# 8. 저장 — 내일 다시 쓰려면
# =====================================================================
import joblib

joblib.dump(
    clf, "고장분류기.joblib"
)  # 파이프라인(스케일러+모델) 통째로 파일로. 학습 결과가 다 들어감
다시 = joblib.load(
    "고장분류기.joblib"
)  # 내일 이 한 줄로 불러오면 재학습 없이 바로 예측
print(
    "\n[8] 저장 후 불러와 예측 — 새 설비 [공기 299, 회전 1400, 토크 55, 마모 240] 고장 확률:",
    round(다시.predict_proba([[299, 1400, 55, 240]])[0, 1], 3),
)
os.remove("고장분류기.joblib")  # 수업 폴더를 깨끗하게 (실제론 남겨 둡니다)


# =====================================================================
# 9. 정리 — 손코드 ↔ 사이킷런 대응표, 그리고 언제 쓰나
# =====================================================================
print("""
[9] 손으로 한 것 ↔ 사이킷런
    01 표준화 (x−평균)/표준편차     ↔  StandardScaler().fit(train) / .transform()
    02 섞어서 70:30 자르기          ↔  train_test_split(X, y, test_size=0.3, random_state=…, stratify=y)
    01·02 경사하강 300·500걸음      ↔  LinearRegression().fit(X, y)   (공식으로 단번에)
    03 sigmoid + 로그손실 + 경사하강 ↔  LogisticRegression().fit(X, y)  (속으로 걸음, max_iter)
    03 확률 / 0.5 판정              ↔  .predict_proba(X)[:, 1] / .predict(X)
    03 네 칸·재현율 손으로 세기      ↔  confusion_matrix / classification_report / recall_score
    w, b                           ↔  .coef_, .intercept_
    (없음) 과적합 처방               ↔  Ridge(alpha) / Lasso(alpha)
    (없음) 불균형 처방               ↔  class_weight="balanced" (+ 임계값 조정)
    (없음) 손잡이 고르기             ↔  cross_val_score / GridSearchCV

    사이킷런은 언제?  표 데이터, 몇백~몇십만 행, 선형모델·트리·부스팅 → 실무 기본. fit 한 줄. 이 과정 데이터는 전부 여기.
    파이토치는 언제?  학습 루프를 내 손으로 쥐어야 할 때 — 아주 큰 데이터, 손실을 내 맘대로 바꿀 때,
                     그리고 다음 과정에서 배울 '신경망'(층 쌓기) → 05 파일에서 같은 문제를 파이토치로 풀어 봅니다.
""")

# =====================================================================
# 실습 — Ridge 말고 Lasso 는?
# =====================================================================
# [문제] 9번 대응표에 Ridge 와 함께 Lasso 가 적혀 있었습니다.
# Lasso(alpha=...) 로 바꿔서 alpha 0.01 / 0.1 / 1.0 의 train, test 점수를 보세요.
#
#   힌트: from sklearn.linear_model import Lasso 부터. 나머지는 6번의 Ridge 코드와 같습니다.
