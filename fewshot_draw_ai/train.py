import numpy as np
import glob
from model import FewShotModel


X=[]
y=[]


files = glob.glob(
"data/train/*.npy"
)


for i,f in enumerate(files):

    img=np.load(f)

    X.append(
        img.flatten()
    )

    # 仮ラベル
    y.append(i%2)



model = FewShotModel()


model.fit(
    np.array(X),
    np.array(y)
)


model.save(
"saved_model/fewshot.pkl"
)


print(
"training finished"
)
