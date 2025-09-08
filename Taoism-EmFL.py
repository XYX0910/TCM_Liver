"""
Taoism-EmFL：基于五行-阴阳理论的改进型情感模糊学习网络
------------------------------------------------------------
已删除原始 EmFL 方法，仅保留所需组件与主流程：
1. 数据读取与常量定义
2. MembershipFunction、NeuroFuzzyNetwork 基类
3. EvolutionaryOptimizer（遗传/迁移+局部搜索）
4. TaoismEmFL（改进模型）
5. 评估与主函数
"""

import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score,
                             precision_score, recall_score, f1_score)
from numpy.linalg import pinv
import matplotlib.pyplot as plt

# ==================== 超参与数据路径 ==================== #
DATA_PATH   = r"C://Users//Xie Yuxuan//Desktop//AI and Liver Diagnosis//data_11250.xlsx"
TEST_RATIO  = 0.2
RANDOM_SEED = 42

NUM_OUTPUTS       = 6          # 6 个证候
RULES_PER_OUTPUT  = 2          # 每个证候 2 条规则（正向）
NUM_RULES         = NUM_OUTPUTS * RULES_PER_OUTPUT

POP_SIZE  = 50
MAX_GEN   = 2
SAMPLE_SIZE = None             # =None 时使用全部数据

LOCAL_NEIGHBORS = 5            # 局部搜索邻居
LOCAL_BETA      = 0.1          # 局部扰动幅度

np.random.seed(RANDOM_SEED)

# 针对 6 个证候的特征子集（与原始 EmFL 保持一致）
FEATURE_SUBSETS = {
    0: [6,14,22,31,39,46,50,52],
    1: [2,4,7,8,9,11,12,16,17,24,26,35,40,43,44,47,50,53],
    2: [1,2,3,4,6,7,9,15,17,24,26,28,29,30,31,35,42,44,48,53],
    3: [0,2,3,6,7,11,14,15,17,18,24,26,27,28,30,31,33,34,36,42,46,48,49,51,53],
    4: [9,10,11,12,13,14,17,19,22,23,25,28,32,38,39,41,45,47,50,52,53],
    5: [0,5,6,17,18,20,21,22,24,25,28,31,32,36,37,42,46,48,51,54,55]
}

# ==================== 数据读取 ==================== #
def load_data(file_path=DATA_PATH, test_ratio=TEST_RATIO,
              sample_size=SAMPLE_SIZE, random_seed=RANDOM_SEED):
    data = pd.read_excel(file_path)
    if sample_size:
        data = data.sample(n=sample_size, random_state=random_seed)
    if data.shape[1] != 62:
        raise ValueError(f"Expect 62 columns, got {data.shape[1]}")

    X = data.iloc[:, :56].values.astype(float)
    y = data.iloc[:, 56:].values.astype(float)

    return train_test_split(X, y, test_size=test_ratio, random_state=random_seed)

# ==================== 隶属函数族 ==================== #
class MembershipFunction:
    EPS = 1e-12
    # ---- 基础形状 ---- #
    @staticmethod
    def _tri(a,b,c,x):
        if a < x <= b: return (x-a)/(b-a+MembershipFunction.EPS)
        if b < x < c : return (c-x)/(c-b+MembershipFunction.EPS)
        return 0.0
    @staticmethod
    def _gauss(mu,sigma,x):
        return np.exp(-((x-mu)**2)/(2*(sigma+MembershipFunction.EPS)**2))
    # ---- 典型 12 类 ---- #
    @staticmethod
    def f1_triangular(x,p):           # a,b,c
        return MembershipFunction._tri(*p,x)
    @staticmethod
    def f2_left_tri(x,p):             # b,c
        b,c=p;  return 1.0 if x<=b else (c-x)/(c-b+MembershipFunction.EPS) if x<c else 0.0
    @staticmethod
    def f3_right_tri(x,p):            # a,b
        a,b=p;  return 1.0 if x>b else MembershipFunction._tri(a,b,b,x)
    @staticmethod
    def f4_trap(x,p):                 # a,b,c,d
        a,b,c,d=p
        if b<=x<=c: return 1.0
        if a<x<b: return (x-a)/(b-a+MembershipFunction.EPS)
        if c<x<d: return (d-x)/(d-c+MembershipFunction.EPS)
        return 0.0
    @staticmethod
    def f5_gauss(x,p):                # mu,sigma
        return MembershipFunction._gauss(*p,x)
    @staticmethod
    def f6_left_gauss(x,p):
        mu,sigma=p; return 1.0 if x<mu else MembershipFunction._gauss(mu,sigma,x)
    @staticmethod
    def f7_right_gauss(x,p):
        mu,sigma=p; return 1.0 if x>mu else MembershipFunction._gauss(mu,sigma,x)
    @staticmethod
    def f8_flat_gauss(x,p):           # a,b,mu,sigma
        a,b,mu,sigma=p; return 1.0 if a<=x<=b else MembershipFunction._gauss(mu,sigma,x)
    @staticmethod
    def f9_pi(x,p):                   # a,b,c   (平滑 π)
        a,b,c=p
        if x<=a or x>=c: return 0.0
        if x<b: return 0.5*(1-np.cos(np.pi*(x-a)/(b-a+MembershipFunction.EPS)))
        return 0.5*(1+np.cos(np.pi*(x-b)/(c-b+MembershipFunction.EPS)))
    @staticmethod
    def f10_left_pi(x,p):             # b,c
        b,c=p
        if x<=b: return 1.0
        if x>=c: return 0.0
        return 0.5*(1+np.cos(np.pi*(x-b)/(c-b+MembershipFunction.EPS)))
    @staticmethod
    def f11_right_pi(x,p):            # a,b
        a,b=p
        if x<=a: return 0.0
        if x>=b: return 1.0
        return 0.5*(1-np.cos(np.pi*(x-a)/(b-a+MembershipFunction.EPS)))
    @staticmethod
    def f12_flat_pi(x,p):             # a,b,c,d
        a,b,c,d=p
        if x<=a or x>=d: return 0.0
        if b<=x<=c: return 1.0
        if x<b: return 0.5*(1-np.cos(np.pi*(x-a)/(b-a+MembershipFunction.EPS)))
        return 0.5*(1+np.cos(np.pi*(x-c)/(d-c+MembershipFunction.EPS)))

    TYPE_MAP = {
        1:f1_triangular.__func__, 2:f2_left_tri.__func__, 3:f3_right_tri.__func__,
        4:f4_trap.__func__, 5:f5_gauss.__func__, 6:f6_left_gauss.__func__,
        7:f7_right_gauss.__func__, 8:f8_flat_gauss.__func__, 9:f9_pi.__func__,
        10:f10_left_pi.__func__, 11:f11_right_pi.__func__, 12:f12_flat_pi.__func__
    }

    @staticmethod
    def param_dim(t:int)->int:
        if t in (1,9): return 3
        if t in (4,8,12): return 4
        return 2

# ==================== 基础 NeuroFuzzyNetwork ==================== #
class NeuroFuzzyNetwork:
    def __init__(self, num_rules=NUM_RULES, num_features=56, num_outputs=NUM_OUTPUTS):
        self.R, self.M, self.O = num_rules, num_features, num_outputs
        self.rules, self.rule_param_index = [], []
        self.Q_matrix, self.P_matrix = None, None
        self._init_rules()

    def _rand_params(self, t:int):
        n = MembershipFunction.param_dim(t)
        if t in (5,6,7):                      # Gaussian
            return np.array([np.random.beta(2,2), np.random.uniform(0.15,0.35)])
        if n==4:                              # 4-param
            pts=np.sort(np.random.rand(4))
            if pts[2]-pts[1]<0.1:             # 展开平台
                mid=0.5*(pts[1]+pts[2]); pts[1]=mid-0.05; pts[2]=mid+0.05
            return pts
        pts=np.sort(np.random.rand(n))        # 2/3-param
        if n==3 and (pts[1]-pts[0]<0.05 or pts[2]-pts[1]<0.05):
            pts[1]=pts[0]+0.05; pts[2]=pts[1]+0.05
        return pts

    def _init_rules(self):
        self.rules.clear(); self.rule_param_index=[]
        start=0
        for r in range(self.R):
            s_idx = r // RULES_PER_OUTPUT
            feats = FEATURE_SUBSETS[s_idx]
            p=len(feats)
            mftypes = np.random.randint(1,13,size=p)
            mfparams=[self._rand_params(t) for t in mftypes]
            self.rules.append({"syndrome_idx":s_idx,"features":feats,
                               "mf_types":mftypes,"mf_params":mfparams})
            self.rule_param_index.append((start,start+p+1)); start+=p+1

    # ---------- 触发强度、建模、预测 ---------- #
    def _firing_strength(self,x):
        F=np.zeros(self.R)
        for r,rule in enumerate(self.rules):
            mvals=[max(MembershipFunction.TYPE_MAP[rule["mf_types"][j]](x[idx],
                    rule["mf_params"][j]),1e-6) for j,idx in enumerate(rule["features"])]
            F[r]=np.prod(mvals)
        return F

    def _build_hidden_matrix(self, F_bar, X):
        """
        构建隐藏矩阵 H，并准确记录每条规则对应的
        参数区间 self.rule_param_index。
        关键：对每条规则都用 rule['features'] 而不是固定子集！
        """
        H_parts = []
        self.rule_param_index = []
        start_idx = 0

        for r in range(self.R):
            rule  = self.rules[r]               # ← 直接取规则
            feats = rule['features']            # ← 真实特征索引
            p     = len(feats)
            length = p + 1                      # 常数项 + p 个系数

            # 记录 [start, end) 供后续切片
            self.rule_param_index.append((start_idx, start_idx + length))
            start_idx += length

            # 构造 H_r = [f_r , f_r * X_sub]
            f_r   = F_bar[:, r:r+1]             # (N,1)
            X_sub = X[:, feats]                 # (N,p)
            H_r   = np.concatenate([f_r, f_r * X_sub], axis=1)
            H_parts.append(H_r)

        # 拼接得到最终 H
        return np.concatenate(H_parts, axis=1)

    def fit(self,X,Y,incremental=False):
        F=np.array([self._firing_strength(x) for x in X])
        Fbar=F/np.maximum(F.sum(axis=1,keepdims=True),1e-12)
        H=self._build_hidden_matrix(Fbar,X)
        Y_exp=np.zeros((len(X),self.R))
        for r in range(self.R): Y_exp[:,r]=Y[:,self.rules[r]["syndrome_idx"]]

        if not incremental or self.P_matrix is None:
            self.Q_matrix=pinv(H)@Y_exp
            self.P_matrix=pinv(H.T@H)
        else:                                   # 增量 RLS
            I=np.eye(H.shape[0])
            S=I+H@self.P_matrix@H.T
            K=self.P_matrix@H.T@pinv(S)
            self.Q_matrix+=K@(Y_exp-H@self.Q_matrix)
            self.P_matrix-=K@H@self.P_matrix

    def predict(self,X):
        pred=np.zeros((len(X),self.O))
        for i,x in enumerate(X):
            f=self._firing_strength(x); s=f/np.maximum(f.sum(),1e-12)
            num=np.zeros(self.O); den=np.zeros(self.O)
            for r,rule in enumerate(self.rules):
                k=rule["syndrome_idx"]
                start,end=self.rule_param_index[r]
                Q_r=self.Q_matrix[start:end,r]
                inp=np.concatenate([[1.0], x[rule["features"]]])
                y_r=inp@Q_r
                num[k]+=s[r]*y_r; den[k]+=s[r]
            pred[i]=np.where(den>1e-6,num/den,0.0)
        return pred

    def load_rule_set(self,rule_set):
        self.rules=copy.deepcopy(rule_set)
        self.rule_param_index=[]; start=0
        for rule in self.rules:
            p=len(rule["features"])
            self.rule_param_index.append((start,start+p+1)); start+=p+1
        self.Q_matrix=None; self.P_matrix=None

# ==================== Evolutionary Optimizer ==================== #
class EvolutionaryOptimizer:
    def __init__(self,pop_size=POP_SIZE,max_gen=MAX_GEN,
                 local_neighbors=LOCAL_NEIGHBORS,beta=LOCAL_BETA,
                 elitism=2,early_stop_patience=10,min_delta=1e-4):
        self.P,self.G=pop_size,max_gen
        self.local_neighbors,self.beta=local_neighbors,beta
        self.elitism,self.early_stop_patience,self.min_delta=elitism,early_stop_patience,min_delta
        self.population=[]

    def _clone_rules(self,rules):
        return [{"mf_types":r["mf_types"].copy(),
                 "mf_params":[p.copy() for p in r["mf_params"]],
                 "syndrome_idx":r["syndrome_idx"],
                 "features":r["features"]} for r in rules]

    def init_population(self,template_net):
        self.population=[]
        for _ in range(self.P):
            indiv=self._clone_rules(template_net.rules)
            for rule in indiv:
                idx=np.random.randint(len(rule["mf_types"]))
                new_t=np.random.randint(1,13)
                rule["mf_types"][idx]=new_t
                rule["mf_params"][idx]=template_net._rand_params(new_t)
            self.population.append(indiv)

    def fitness(self,indiv,net,X,Y):
        net.load_rule_set(indiv); net.fit(X,Y)
        pred=net.predict(X); n=len(Y)
        rmse=np.sqrt(np.sum((Y-pred)**2)/(64*n))
        return 1.0-rmse

    def migrate(self,fitness_vals):
        f_max,f_min=np.max(fitness_vals),np.min(fitness_vals); eps=1e-6
        em_probs=(fitness_vals-f_min+eps)/(f_max-f_min+eps); em_probs/=em_probs.sum()
        new_pop=[]
        for i,indiv in enumerate(self.population):
            imm_prob=(f_max-fitness_vals[i]+eps)/(f_max-f_min+eps)
            rule_set=copy.deepcopy(indiv)
            for r in range(len(rule_set)):
                for j in range(len(rule_set[r]['mf_types'])):
                    if np.random.rand()<imm_prob:
                        donor_idx=np.random.choice(self.P,p=em_probs)
                        donor=self.population[donor_idx]
                        donor_rule=donor[r]
                        if rule_set[r]['mf_types'][j]==donor_rule['mf_types'][j] and \
                           len(rule_set[r]['mf_params'][j])==len(donor_rule['mf_params'][j]):
                            cur,don=rule_set[r]['mf_params'][j],donor_rule['mf_params'][j]
                            rule_set[r]['mf_params'][j]=cur+np.random.rand()*(don-cur)
                        else:
                            rule_set[r]['mf_types'][j]=donor_rule['mf_types'][j]
                            scale=(f_max-fitness_vals[donor_idx]+eps)/(f_max-f_min+eps)
                            rule_set[r]['mf_params'][j]=np.array(
                                [val+np.random.uniform(-1,1)*scale for val in donor_rule['mf_params'][j]])
            new_pop.append(rule_set)
        return new_pop

    def optimise(self,net,X,Y):
        self.init_population(net); best_fit=-np.inf; best_indiv=None; patience=0
        for g in range(self.G):
            fitness_vals=np.array([self.fitness(ind,net,X,Y) for ind in self.population])
            elite_idx=fitness_vals.argsort()[-self.elitism:]; elites=[self._clone_rules(self.population[i]) for i in elite_idx]
            gen_best=fitness_vals[elite_idx[-1]]
            if gen_best>best_fit+self.min_delta:
                best_fit, best_indiv = gen_best, self._clone_rules(elites[-1]); patience=0
                # 局部搜索
                for _ in range(self.local_neighbors):
                    neigh=self._clone_rules(best_indiv)
                    rr=np.random.randint(len(neigh)); jj=np.random.randint(len(neigh[rr]['mf_params']))
                    params=neigh[rr]['mf_params'][jj].copy()
                    kk=np.random.randint(params.size); params[kk]+=np.random.randn()*self.beta
                    neigh[rr]['mf_params'][jj]=params
                    fit_n=self.fitness(neigh,net,X,Y)
                    if fit_n>best_fit: best_fit, best_indiv=self.fitness(neigh,net,X,Y),self._clone_rules(neigh)
            else: patience+=1
            print(f"Gen {g:02d}/{self.G} | gen_best={gen_best:.4f} | global_best={best_fit:.4f} | patience={patience}")
            if patience>=self.early_stop_patience:
                print(f"Early stop at gen {g:02d}"); break
            migrants=self.migrate(fitness_vals)
            self.population=elites+migrants
            self.population=self.population[:self.P]
        net.load_rule_set(best_indiv); net.fit(X,Y)
        print(f"Optimisation done. Best fitness={best_fit:.4f}")
        return net

# ==================== 评估辅助函数 ==================== #
def classification_report(y_true,y_pred):
    yt=np.where(y_true<1.5,0,np.where(y_true<4.5,1,2))
    yp=np.where(y_pred<1.5,0,np.where(y_pred<4.5,1,2))
    for j in range(yt.shape[1]):
        print(f"--- Syndrome {j} ---")
        for cls,label in zip([0,1,2],['None','Mild','Severe']):
            yj_t=(yt[:,j]==cls).astype(int); yj_p=(yp[:,j]==cls).astype(int)
            p=precision_score(yj_t,yj_p,zero_division=0)
            r=recall_score(yj_t,yj_p,zero_division=0)
            f=f1_score(yj_t,yj_p,zero_division=0)
            print(f"{label:>6s} | P={p:.3f} R={r:.3f} F1={f:.3f}")

# ==================== Taoism-EmFL ==================== #
class TaoismEmFL(NeuroFuzzyNetwork):
    def __init__(self, num_rules=RULES_PER_OUTPUT, num_features=56):
        super().__init__(num_rules, num_features)

        # ---- 五行分组 (示例) ----
        self.wuxing_groups = {
            'wood':  [7, 15, 23, 32, 40, 47],
            'fire':  [1, 3, 4, 7, 8, 12, 15, 16],
            'earth': [20, 21, 22, 23, 26],
            'metal': [11, 12, 13, 14],
            'water': [10, 24, 30, 41]
        }

        # ---- 阴阳症状表 ----
        self.yin_symptoms  = [10, 11, 13, 20, 24, 26, 42, 46, 48, 51]
        self.yang_symptoms = [1, 3, 4, 12, 15, 18, 19, 25, 27, 37, 43, 49, 52]

        self.feature_weights    = np.ones(num_features)
        self.yin_yang_balance   = 0.5  # 初始阴阳平衡

    # ---------- 动态权重 & 阴阳隶属 ---------- #
    def update_dynamic_weights(self, x):
        """根据五行强度实时更新特征权重与阴阳平衡"""
        self.feature_weights = np.ones(self.M)

        # 1) 计算五行组平均强度
        grp_strength = {e: np.mean([x[i] for i in idx if i < self.M])
                        for e, idx in self.wuxing_groups.items()}

        # 2) 木生火、木克土
        for idx in self.wuxing_groups['fire']:
            if idx < self.M:
                self.feature_weights[idx] = 1.0 + 0.5 * grp_strength['wood']
        for idx in self.wuxing_groups['earth']:
            if idx < self.M:
                self.feature_weights[idx] = 1.0 - 0.3 * grp_strength['wood']

        # 3) 阴阳平衡
        yin_strength  = np.mean([x[i] for i in self.yin_symptoms  if i < self.M]) if self.yin_symptoms  else 0
        yang_strength = np.mean([x[i] for i in self.yang_symptoms if i < self.M]) if self.yang_symptoms else 0
        total = yin_strength + yang_strength
        self.yin_yang_balance = yang_strength / total if total > 0 else 0.5

    def yin_yang_membership(self, x, params):
        """互补式阴阳隶属度（最多四个参数，高斯对）"""
        yin  = MembershipFunction._gauss(params[0], params[1], x) if len(params) >= 2 else 0.0
        yang = MembershipFunction._gauss(params[2], params[3], x) if len(params) >= 4 else 0.0
        tot = yin + yang + 1e-6
        return yin / tot, yang / tot

    # 🟢 修正：仅遍历该规则真正拥有的特征，避免越界
    def compute_membership(self, x, rule_idx):
        self.update_dynamic_weights(x)

        rule = self.rules[rule_idx]
        strength = 1.0
        for j, feat_idx in enumerate(rule['features']):   # 只遍历子特征
            weighted_x = x[feat_idx] * self.feature_weights[feat_idx]
            yin, yang  = self.yin_yang_membership(weighted_x, rule['mf_params'][j])
            combined   = yin * (1 - self.yin_yang_balance) + yang * self.yin_yang_balance
            strength  *= combined                        # 乘积合成
        return strength

    def _firing_strength(self, x):
        return np.array([self.compute_membership(x, r) for r in range(self.R)])

    # ---------- 证候逻辑约束 ---------- #
    def apply_constraints(self, out):
        if len(out) < 6:
            out = np.zeros(6)

        # 肝火(3) vs 肝寒(4) 互斥
        if out[3] > 4.0 and out[4] > 4.0:
            if out[3] > out[4]:
                out[4] = max(0.0, out[4] - 2.0)
            else:
                out[3] = max(0.0, out[3] - 2.0)

        # 肝郁(0) 推升肝火(3)
        if out[0] > 4.0:
            out[3] = min(8.0, out[3] + 0.5 * (out[0] - 4.0))

        # 肝阴虚(2) 推升肝火(3)
        if out[2] > 4.0:
            out[3] = min(8.0, out[3] + 0.3 * (out[2] - 4.0))

        return out

    def predict(self, X):
        raw = super().predict(X)
        return np.array([self.apply_constraints(o) for o in raw])

    def add_negative_rules(self):
        """为每个证候补充否定规则"""
        for k in range(6):
            neg = {
                'mf_types':  np.random.randint(1, 13, size=self.M),
                'mf_params': [np.random.rand(4) for _ in range(self.M)],
                'is_negative': True,
                'syndrome_idx': k,
                'features': list(range(self.M))
            }
            self.rules.append(neg)
        self.R = len(self.rules)

# ==================== 训练与测试流程 ==================== #
def train_and_evaluate_taoism(sample_size=SAMPLE_SIZE):
    X_tr,X_te,y_tr,y_te=load_data(sample_size=sample_size)
    model=TaoismEmFL(num_rules=RULES_PER_OUTPUT,num_features=56)
    model._init_rules()
    model.add_negative_rules()

    optimizer=EvolutionaryOptimizer(pop_size=POP_SIZE,max_gen=MAX_GEN)
    model=optimizer.optimise(model,X_tr,y_tr)

    def _eval(m,X,y,name):
        p=m.predict(X)
        print(f"[{name}] MAE={mean_absolute_error(y,p):.3f} "
              f"MSE={mean_squared_error(y,p):.3f} "
              f"R²={r2_score(y,p):.3f} "
              f"EVS={explained_variance_score(y,p):.3f}")
        return p

    _= _eval(model,X_tr,y_tr,"TRAIN")
    preds= _eval(model,X_te,y_te,"TEST")

    print("\n===== Taoism-EmFL Classification Report (TEST) =====")
    classification_report(y_te,preds)

    # 阴阳平衡分布
    balances=[(model.update_dynamic_weights(x),model.yin_yang_balance)[1] for x in X_te]
    plt.figure(figsize=(7,5))
    plt.hist(balances,bins=20)
    plt.xlabel("Yin-Yang Balance Index"); plt.ylabel("Frequency")
    plt.title("Distribution on Test Set"); plt.tight_layout(); plt.show()

# ==================== 主入口 ==================== #
if __name__=="__main__":
    print(">> Training and Testing Improved Taoism-EmFL ...")
    train_and_evaluate_taoism()