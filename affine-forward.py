import numpy as np


def affine_forward(
    x: np.array, w: np.array, b: np.array
) -> tuple[np.array, tuple[np.array]]:

    x_reshaped = x.reshape(x.shape[0], -1)

    assert (
        x_reshaped.shape[-1] == w.shape[0]
    ), f"can't do matmul since x_reshaped.shape[-1] (which is {
        x_reshaped.shape[-1]}) and w.shape[-2] (which is {w.shape[-2]
                                                          }) are not equal."
    return x_reshaped @ w, (x_reshaped, w, b)


if __name__ == "__main__":
    x = np.random.rand(40, 3, 4, 5)
    w = np.random.rand(60, 15)
    b = np.random.rand(15)

    affine_F = affine_forward(x, w, b)
    z, (x, w, b) = affine_F
    print(
        f"""post-affine transformation shape (i.e., xW + b): {z.shape}
x.shape after flattening: {x.shape}
w.shape: {w.shape}
b.shape: {b.shape}
          """
    )
