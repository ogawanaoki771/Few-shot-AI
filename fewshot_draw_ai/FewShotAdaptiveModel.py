import numpy as np
import random
import copy


# =========================================================
# Model Config
# =========================================================

IMG_SIZE = 64

MAX_CELLS = 105
MIN_CELLS = 10

INITIAL_CELLS = 15

TRAIN_STEPS = 28
TEST_TTA_STEPS = 28


SEED = 2026

random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)



# =========================================================
# Line Adaptive Few-shot Cell
# =========================================================

class LineAdaptiveCell:

    def __init__(self, cell_id):

        self.id = cell_id


        # 64x64対応
        self.anchor_x = rng.uniform(
            8,
            IMG_SIZE-8
        )

        self.anchor_y = rng.uniform(
            8,
            IMG_SIZE-8
        )


        self.geom_sensitivity = rng.uniform(
            1.0,
            5.0
        )


        self.learning_rate = rng.uniform(
            0.05,
            0.35
        )


        self.energy = rng.uniform(
            0.85,
            1.0
        )


        self.links = {}


        self.activation_history = [
            0.0
        ] * 5


        self.last_activation = 0.0



    def reset(self):

        self.activation_history = [
            0.0
        ] * 5

        self.last_activation = 0.0



    def activate(
        self,
        coords
    ):


        if len(coords)==0:
            return 0.0


        pts=np.asarray(
            coords,
            dtype=float
        )


        distance=np.sqrt(
            (
                pts[:,0]
                -
                self.anchor_x
            )**2
            +
            (
                pts[:,1]
                -
                self.anchor_y
            )**2
        )


        near = np.sum(
            distance < 12
        )


        response=np.tanh(
            near
            /
            (
                self.geom_sensitivity
                *8
                +1e-9
            )
        )


        self.last_activation=float(
            response
        )


        self.activation_history.append(
            self.last_activation
        )


        if len(self.activation_history)>5:
            self.activation_history.pop(0)


        return self.last_activation



    def adapt(
        self,
        coords
    ):

        if len(coords)==0:
            return


        pts=np.asarray(
            coords,
            dtype=float
        )


        mean=np.mean(
            pts,
            axis=0
        )


        self.anchor_x += (
            self.learning_rate
            *
            (
                mean[0]
                -
                self.anchor_x
            )
        )


        self.anchor_y += (
            self.learning_rate
            *
            (
                mean[1]
                -
                self.anchor_y
            )
        )


        self.anchor_x=np.clip(
            self.anchor_x,
            0,
            IMG_SIZE
        )


        self.anchor_y=np.clip(
            self.anchor_y,
            0,
            IMG_SIZE
        )




# =========================================================
# Few-shot Ecosystem Model
# =========================================================

class FewShotAdaptiveModel:


    def __init__(
        self,
        cells=INITIAL_CELLS
    ):

        self.pool=[
            LineAdaptiveCell(i)
            for i in range(cells)
        ]



    def reset_dynamic(self):

        for c in self.pool:
            c.reset()



    def encode(
        self,
        coords,
        steps=28
    ):


        self.reset_dynamic()


        features=[]


        for t in range(steps):

            acts=[]


            for cell in self.pool:

                a=cell.activate(
                    coords
                )

                acts.append(a)



            features.extend(
                [
                    np.mean(acts),
                    np.max(acts),
                    np.std(acts)
                ]
            )


            for cell in self.pool:

                cell.adapt(
                    coords
                )


        return np.asarray(
            features,
            dtype=float
        )



    def fit(
        self,
        X_train,
        y_train
    ):


        self.memory=[]


        for x,y in zip(
            X_train,
            y_train
        ):

            feature=self.encode(
                x,
                TRAIN_STEPS
            )


            self.memory.append(
                (
                    feature,
                    y
                )
            )



    def predict(
        self,
        X_test
    ):


        predictions=[]


        for x in X_test:


            feature=self.encode(
                x,
                TEST_TTA_STEPS
            )


            distances=[]


            for train_feature,label in self.memory:

                d=np.linalg.norm(
                    feature
                    -
                    train_feature
                )

                distances.append(
                    (
                        d,
                        label
                    )
                )


            distances.sort(
                key=lambda z:z[0]
            )


            # 最近傍Few-shot
            pred=distances[0][1]


            predictions.append(
                pred
            )


        return np.array(
            predictions
        )
