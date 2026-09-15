# =====================================================================
#  실습 2 — 진동으로 압연 모터 전류를 맞혀 보기 (02 다변수 선형회귀)
# =====================================================================
#  실행: python 실습2_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas  /  데이터는 옆의 데이터/ 폴더에 들어 있습니다.
#
#  [상황]
#    P제철 열간압연기(SPM01)에 진동센서 두 개(TOP·BOT)와 모터 전류계가 붙어 있습니다.
#    그런데 전류계가 자주 고장 납니다. 진동만 있을 때 전류를 추정할 수 있을까요?
#    맞힐 대상(정답) = CUR-MTR_RMS  (모터 전류의 실효값)
#
#  [쓰는 도구]  02 에서 배운 것 전부. 새 라이브러리 없습니다.
#    다변수 X / train·test 분할 / 열별 표준화(학습용 통계로만) / 경사하강 / R2
#    ※ 02_선형회귀_다변수_train_test.py 를 옆에 띄워 놓고 베껴 쓰세요. 그게 정상입니다.
#
#  [푸는 법]  TODO 를 위에서부터 하나씩 채우고, 그때그때 실행해서 숫자를 확인하세요.
#             한 번에 다 짜고 실행하면 어디서 틀렸는지 못 찾습니다.
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.


# =====================================================================
# A. 데이터부터 본다  (모델 얘기는 아직 이르다)
# =====================================================================
# [A1] 표의 모양과 열 이름, 결측 개수를 찍으세요.
#      힌트: d.shape / list(d.columns) / d.isna().sum().sum()
# TODO
print("[A1]")
print("표의 모양:", d.shape)
print("열 이름:", list(d.columns))
print("결측 개수:", d.isna().sum().sum())


# [A2] 정답으로 쓸 CUR-MTR_RMS 의 요약통계를 보세요. (describe)
#      → 이 값이 대략 몇에서 몇 사이인지 말할 수 있어야 합니다.
#        나중에 "MSE 500" 이 큰 건지 작은 건지 판단하는 기준이 됩니다.
# TODO
print("\n[A2] CUR-MTR_RMS 요약통계")
print(d["CUR-MTR_RMS"].describe())

# CUR-MTR_RMS는 약 6.33에서 193.42 사이이고, 평균은 약 111.29입니다.


# [A3] 숫자 열들의 상관계수 중, CUR-MTR_RMS 와의 상관만 크기순으로 보세요.
#      힌트: d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values()
#
#      ★ 보고 나서 답하세요 (주석으로 적어 두기) ★
#        (1) 상관이 0.98, 0.99 로 말도 안 되게 높은 열이 몇 개 보입니다. 이름이 뭔가요?
#        (2) 그 열들을 입력으로 쓰면 안 되는 이유가 있습니다. 뭘까요?
#            힌트: 열 이름의 앞부분을 보세요. CUR-MTR-... 로 시작하죠.
#                 전류계가 고장 나서 전류를 추정하려는 건데, 그 입력은 어디서 옵니까?
#      내 답: (1)
#             (2)
# TODO
print("\n[A3] CUR-MTR_RMS와의 상관계수")
cur_corr = d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values(ascending=False)
print(cur_corr)

# 내 답: (1) 정답 자신을 제외하면 CUR-MTR_STD(약 0.999)와
#            CUR-MTR_PTP(약 0.982)의 상관이 비정상적으로 높습니다.
#         (2) 두 열 역시 고장 날 수 있는 같은 전류계의 측정값으로부터 만들어집니다.
#            전류계가 고장 나 RMS를 측정할 수 없는 상황에는 STD와 PTP도 얻을 수 없으므로,
#            이 열들을 입력으로 사용하면 실제 배포 시 사용할 수 없는 정보가 새어 들어갑니다.


# =====================================================================
# B. 입력 고르고 train / test 나누기
# =====================================================================
# [B1] 입력(특징) 4개를 아래 이름 그대로 쓰세요. 정답은 CUR-MTR_RMS.
특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS"]
# X = ...   (d[특징이름].values.astype(float))
# y = ...   (d["CUR-MTR_RMS"].values.astype(float))
# X.shape, y.shape 를 찍어서 (570, 4) 와 (570,) 인지 확인하세요.
# TODO
X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)
print(f"\n[B1] X.shape: {X.shape} / y.shape: {y.shape}")


# [B2] 7:3 으로 나누세요. 반드시 '섞은 다음에' 자릅니다.
#      RandomState(42) 를 쓰면 정답지와 숫자가 똑같이 나옵니다.
#      힌트: 순서 = np.random.RandomState(42).permutation(len(X))
#            n_train = int(len(X) * 0.7)
#      학습용 몇 대 / 시험용 몇 대인지 찍으세요.
# TODO
order = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)
tr = order[:n_train]
te = order[n_train:]

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]
print(f"\n[B2] 학습용 {len(X_train)}대 / 시험용 {len(X_test)}대")


# [B3] 열별 표준화. ★ mu 와 sd 는 학습용에서만 구합니다 ★
#      시험용도 학습용의 mu, sd 로 변환하세요.
#      확인: 표준화 후 학습용 열별 평균은 0, 퍼짐은 1.
#            시험용 평균은 0 이 아닙니다. 그게 맞습니다 (이유를 말할 수 있어야 합니다).
# TODO
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

print("\n[B3] 표준화 후 학습용 열별 평균:", Z_train.mean(axis=0).round(3))
print("표준화 후 학습용 열별 표준편차:", Z_train.std(axis=0).round(3))
print("표준화 후 시험용 열별 평균:", Z_test.mean(axis=0).round(3))

# 시험용 평균이 0이 아닌 것이 정상입니다. 데이터 누수를 막기 위해 시험용 자체의
# 평균과 표준편차가 아니라 학습용에서 구한 mu와 sd로 시험용을 변환했기 때문입니다.


# =====================================================================
# C. 학습 — 02 의 함수를 그대로 가져다 쓰세요
# =====================================================================
# [C1] 예측 / 손실 / 기울기_밟아보기 / 학습 / R2 / MSE 를 02 에서 복사해 오세요.
#      한 글자도 안 바꿔도 됩니다. 그게 이 실습의 포인트입니다.
#      (데이터가 바뀌어도 걷는 방법은 안 바뀝니다)
# TODO
def 예측(Z, w, b):
    return Z @ w + b


def 손실(Z, y, w, b):
    return np.mean((y - 예측(Z, w, b)) ** 2)


h = 0.0001


def 기울기_밟아보기(Z, y, w, b):
    gw = np.zeros(len(w))
    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h
        w_minus[j] -= h
        gw[j] = (손실(Z, y, w_plus, b) - 손실(Z, y, w_minus, b)) / (2 * h)

    gb = (손실(Z, y, w, b + h) - 손실(Z, y, w, b - h)) / (2 * h)
    return gw, gb


def 학습(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(epochs):
        gw, gb = 기울기_밟아보기(Z, y, w, b)
        w = w - lr * gw
        b = b - lr * gb
    return w, b


def MSE(y, yhat):
    return np.mean((y - yhat) ** 2)


def R2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# [C2] 학습용으로 학습(lr=0.1, epochs=500)하고, 특징별 가중치와 절편 b 를 찍으세요.
# TODO
w, b = 학습(Z_train, y_train, lr=0.1, epochs=500)
print("\n[C2] 특징별 가중치")
for name, wi in zip(특징이름, w):
    print(f"{name:12s}: {wi:+.4f}")
print(f"절편 b       : {b:.4f}")


# [C3] 학습용 R2 / 시험용 R2 를 나란히 찍으세요. MSE 도 같이.
#
#      ★ 답하세요 ★
#        차이가 얼마입니까? 02 의 경보선(0.05 정상 / 0.10 넘으면 의심)에 비춰 보면
#        이 모델은 건강한가요, 과적합인가요?
#        ai4i 데이터(02)에서는 차이가 0.03 이었습니다. 왜 여기선 다를까요?
#      내 답:
# TODO
tr_pred = 예측(Z_train, w, b)
te_pred = 예측(Z_test, w, b)
train_r2 = R2(y_train, tr_pred)
test_r2 = R2(y_test, te_pred)
train_mse = MSE(y_train, tr_pred)
test_mse = MSE(y_test, te_pred)
r2_gap = abs(train_r2 - test_r2)

print("\n[C3]")
print(f"학습용 R2: {train_r2:.4f} / 시험용 R2: {test_r2:.4f}")
print(f"학습용 MSE: {train_mse:.4f} / 시험용 MSE: {test_mse:.4f}")
print(f"R2 차이: {r2_gap:.4f}")

# 내 답: R2 차이는 약 0.1903입니다. 0.10 경보선을 넘으므로 건강한 모델이라기보다
#         과적합을 의심해야 합니다. 학습 데이터의 관계를 시험 데이터에 충분히 일반화하지
#         못했고, 시험 MSE도 학습 MSE보다 큽니다.
#         ai4i보다 차이가 큰 이유는 이 압연 데이터에서 진동과 전류의 관계가 운전 시기·조건에
#         따라 더 달라질 수 있고, 현재 네 특징만으로 그 변화를 충분히 설명하지 못하기 때문입니다.


# =====================================================================
# D. 함정 1 — "점수가 너무 좋으면 의심하라"
# =====================================================================
# [D1] 특징에 "CUR-MTR_STD" 를 하나 추가해서(총 5개) 다시 학습하고 채점하세요.
#      B2 의 분할(순서)은 그대로 재사용합니다. 표준화는 다시 해야 합니다(열이 5개니까).
#
#      ★ 답하세요 ★
#        (1) R2 가 몇으로 나왔나요? 학습용·시험용 둘 다 적으세요.
#        (2) 이 모델을 현장에 넣으면 잘 될까요? 이유는?
#        (3) 이걸 부르는 이름이 있습니다 — '누수(leakage)'.
#            이 경우 정확히 무엇이 새어 들어온 겁니까?
#      내 답: (1)
#             (2)
#             (3)
# TODO
leak_features = 특징이름 + ["CUR-MTR_STD"]
X_leak = d[leak_features].values.astype(float)
X_leak_train, X_leak_test = X_leak[tr], X_leak[te]

leak_mu = X_leak_train.mean(axis=0)
leak_sd = X_leak_train.std(axis=0)
Z_leak_train = (X_leak_train - leak_mu) / leak_sd
Z_leak_test = (X_leak_test - leak_mu) / leak_sd

w_leak, b_leak = 학습(Z_leak_train, y_train, lr=0.1, epochs=500)
leak_tr_pred = 예측(Z_leak_train, w_leak, b_leak)
leak_te_pred = 예측(Z_leak_test, w_leak, b_leak)
leak_train_r2 = R2(y_train, leak_tr_pred)
leak_test_r2 = R2(y_test, leak_te_pred)

print("\n[D1] CUR-MTR_STD를 추가한 모델")
print(f"학습용 R2: {leak_train_r2:.4f} / 시험용 R2: {leak_test_r2:.4f}")

# 내 답: (1) 학습용 R2는 0.9984, 시험용 R2는 0.9982입니다.
#         (2) 현장에 넣으면 안 됩니다. 전류계 고장 시 CUR-MTR_STD도 얻을 수 없습니다.
#         (3) 정답인 전류 RMS와 거의 같은 전류계 신호로 계산한 STD 정보가 입력으로
#             새어 들어온 데이터 누수입니다.


# =====================================================================
# E. 함정 2 — 상관 순위와 실제 쓸모는 다르다
# =====================================================================
# [E1] 02 §6-1 처럼 특징을 하나씩 빼고 다시 학습해, 학습용·시험용 R2 를 각각 찍으세요.
#      (4개 특징이니 4줄이 나옵니다. 힌트: Z_train[:, 남길])
#
#      ★ 먼저 예상하고 적으세요. 실행은 그다음에. ★
#        A3 에서 본 상관을 보면 VIB-BOT_RMS 가 0.74 로 1등,
#        VIB-TOP_RMS 는 0.09 로 사실상 무관해 보입니다.
#        그럼 VIB-TOP_RMS 를 빼도 점수가 안 변하겠죠?
#      내 예상:
# TODO
# 내 예상: 단순 상관만 보면 VIB-BOT_RMS를 빼면 점수가 가장 많이 낮아지고,
#          VIB-TOP_RMS를 빼면 점수가 거의 변하지 않을 것 같습니다.

print("\n[E1] 특징을 하나씩 뺀 결과")
drop_results = {}
for drop_index, drop_feature in enumerate(특징이름):
    keep_indices = [j for j in range(len(특징이름)) if j != drop_index]
    w_drop, b_drop = 학습(Z_train[:, keep_indices], y_train, lr=0.1, epochs=500)
    drop_train_r2 = R2(y_train, 예측(Z_train[:, keep_indices], w_drop, b_drop))
    drop_test_r2 = R2(y_test, 예측(Z_test[:, keep_indices], w_drop, b_drop))
    drop_results[drop_feature] = (drop_train_r2, drop_test_r2)
    print(
        f"{drop_feature:12s} 제외 → "
        f"학습 R2 {drop_train_r2:.4f} / 시험 R2 {drop_test_r2:.4f}"
    )


# [E2] 실행 결과를 보고 답하세요.
#        (1) 예상이 맞았나요?
#        (2) VIB-BOT_RMS 를 뺐을 때 시험용 점수가 어떻게 됐습니까?
#            왜 그럴까요?  힌트: d[특징이름].corr() 를 찍어 보세요.
#                                 VIB-BOT_RMS 와 VIB-BOT_PTP 의 상관은?
#        (3) VIB-TOP_RMS 를 뺐을 때는요? 정답과 상관이 0.09 밖에 안 되는데 왜?
#        (4) 여기서 얻을 교훈을 한 줄로 적으세요.
#      내 답:
# TODO
feature_corr = d[특징이름].corr()
print("\n[E2] 네 입력 특징의 상관계수")
print(feature_corr.round(3))
print(
    "VIB-BOT_RMS와 VIB-BOT_PTP의 상관:",
    round(feature_corr.loc["VIB-BOT_RMS", "VIB-BOT_PTP"], 3),
)

# 내 답: (1) 예상과 달랐습니다.
#         (2) VIB-BOT_RMS를 빼자 시험 R2가 0.6080에서 0.6840으로 오히려 올랐습니다.
#             VIB-BOT_PTP와 상관이 0.953이라 정보가 크게 중복되고, RMS가 시험 데이터에서
#             일반화되지 않는 잡음까지 더했을 가능성이 있습니다.
#         (3) VIB-TOP_RMS를 빼면 시험 R2가 0.3392로 크게 낮아졌습니다. 정답과의 단순
#             상관은 낮아도 다른 특징들을 함께 고려할 때는 보완 정보를 제공하기 때문입니다.
#         (4) 특징의 쓸모는 정답과의 단순 상관만으로 판단하지 말고, 다른 특징과 함께 넣거나
#             뺀 뒤 시험 데이터의 성능으로 판단해야 합니다.


# =====================================================================
# F. 함정 3 — 섞어서 자른 게 정말 옳았나
# =====================================================================
# [F1] 이 데이터는 MEAS_DT(측정시각) 순으로 정렬돼 있습니다. 1월부터 12월까지.
#      02 에서는 "섞고 잘라라, 안 섞으면 편향된다" 고 배웠죠.
#      이번엔 반대로 해 보세요 — 섞지 말고 앞 399행을 학습용, 뒤 171행을 시험용으로.
#      (즉 1~9월로 배워서 10~12월을 맞히기)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 섞었을 때(C3)보다 높나요, 낮나요?
#        (2) 결과가 예상과 다를 겁니다. 그래도 실무에서 예측 모델을 만들 때는
#            보통 이 '시간순 분할' 쪽을 씁니다. 왜 그럴까요?
#            힌트: 현장에 배포된 모델이 맞혀야 하는 데이터는 '언제' 것입니까?
#      내 답: (1)
#             (2)
# TODO
X_time_train, X_time_test = X[:399], X[399:]
y_time_train, y_time_test = y[:399], y[399:]
time_mu = X_time_train.mean(axis=0)
time_sd = X_time_train.std(axis=0)
Z_time_train = (X_time_train - time_mu) / time_sd
Z_time_test = (X_time_test - time_mu) / time_sd

w_time, b_time = 학습(Z_time_train, y_time_train, lr=0.1, epochs=500)
time_train_r2 = R2(y_time_train, 예측(Z_time_train, w_time, b_time))
time_test_r2 = R2(y_time_test, 예측(Z_time_test, w_time, b_time))
print("\n[F1] 시간순 분할")
print(f"학습용 R2: {time_train_r2:.4f} / 시험용 R2: {time_test_r2:.4f}")

# 내 답: (1) 시간순 분할의 시험 R2는 0.7754로, 섞었을 때의 0.6080보다 높습니다.
#         (2) 점수가 우연히 높거나 낮은 것과 별개로, 배포된 모델은 과거로 학습해 미래를
#             예측합니다. 시간순 분할은 미래 정보가 학습에 섞이는 것을 막고 실제 배포 상황과
#             시간에 따른 설비·공정 변화를 더 현실적으로 재현합니다.


# =====================================================================
# G. 데이터가 몇 대면 충분한가
# =====================================================================
# [G1] 학습용을 5 / 10 / 30 / 100 / 399 대로 바꿔 가며 학습하고,
#      학습용 R2 와 시험용 R2 를 표처럼 찍으세요. (적은 데이터는 epochs 를 늘리세요)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 음수로 나오는 구간이 있습니다. 음수는 무슨 뜻입니까?
#            힌트: R2 = 0 이 '무조건 평균만 대답하는 모델' 입니다.
#        (2) 데이터가 늘 때 학습용 점수와 시험용 점수는 각각 어느 방향으로 움직입니까?
#        (3) 이 설비에서 쓸 만한 모델을 만들려면 최소 몇 건쯤 필요해 보입니까?
#      내 답:
# TODO
print("\n[G1] 학습 데이터 수에 따른 성능")
print("학습 건수  학습 R2  시험 R2")
sample_size_results = {}
for n in [5, 10, 30, 100, 399]:
    X_small_train = X_train[:n]
    y_small_train = y_train[:n]
    small_mu = X_small_train.mean(axis=0)
    small_sd = X_small_train.std(axis=0)
    Z_small_train = (X_small_train - small_mu) / small_sd
    Z_small_test = (X_test - small_mu) / small_sd

    w_n, b_n = 학습(Z_small_train, y_small_train, lr=0.1, epochs=2000)
    small_train_r2 = R2(y_small_train, 예측(Z_small_train, w_n, b_n))
    small_test_r2 = R2(y_test, 예측(Z_small_test, w_n, b_n))
    sample_size_results[n] = (small_train_r2, small_test_r2)
    print(f"{n:9d}  {small_train_r2:7.4f}  {small_test_r2:7.4f}")

# 내 답: (1) 시험 R2가 음수라는 것은 시험 데이터의 평균만 항상 예측하는 것보다도
#             성능이 나쁘다는 뜻입니다.
#         (2) 데이터가 늘면 학습 R2는 낮아지지만 더 현실적인 값으로 안정되고, 시험 R2는
#             작은 표본에서 크게 흔들리다가 대체로 높아지며 일반화 격차가 줄어듭니다.
#         (3) 100건의 시험 R2는 약 0.249로 아직 낮습니다. 이 결과만 보면 최소 수백 건,
#             현재 실험에서는 약 399건은 있어야 시험 R2 약 0.607 수준에 도달합니다.


# =====================================================================
# H. 마무리 — 보고서 3줄
# =====================================================================
# 팀장에게 보고한다고 치고, 아래 세 줄을 채우세요.
#
#   1) 전류계가 고장 났을 때 진동으로 전류를 추정할 수 있는가? (된다/안 된다/조건부)
#      근거 점수:
#
#   2) 이 모델을 쓸 때 반드시 붙여야 할 경고 문구 한 줄:
#
#   3) 점수를 더 올리려면 다음에 뭘 해 보겠는가? (한 가지만, 이유와 함께)
#
# =====================================================================

# 1) 조건부로 가능합니다.
#    근거 점수: 누수 없는 네 진동 특징 모델은 무작위 시험 R2 0.6080,
#               실제 배포에 가까운 시간순 시험 R2 0.7754입니다.
#
# 2) 이 추정값은 전류계의 안전 대체값이 아니며, 학습 범위를 벗어난 운전 조건에서는
#    오차가 커질 수 있으므로 경보·정지 판단에는 별도 검증과 안전장치가 필요합니다.
#
# 3) 여러 계절과 운전 조건의 시간순 데이터를 더 수집하겠습니다. 학습량 실험에서
#    적은 데이터의 시험 성능이 매우 불안정했고, 수백 건에서야 일반화 성능이 나아졌기 때문입니다.
