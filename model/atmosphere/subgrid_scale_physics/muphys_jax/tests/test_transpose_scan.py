import jax
import jax.numpy as jnp
from jax import lax

def fun(carry, x):
    new_carry = carry + x
    return new_carry, carry

@jax.jit
def scan(ins):
    carry, outputs = lax.scan(
        fun,
        jnp.zeros((n,)),     # initial carry
        ins,   # inputs
    )
    return outputs

@jax.jit
def transpose_scan(ins):
    ins_t = jnp.transpose(ins)
    carry, outputs = lax.scan(
        fun,
        jnp.zeros((n,)),     # initial carry
        ins_t,   # inputs
    )
    outputs_t = jnp.transpose(outputs)
    return outputs_t

if __name__ == "__main__":
    n = 10
    ins = jnp.ones((n,n))  # Example input
    print("\ninput\n", ins)

    # scan no transpose
    outputs = scan(ins)
    print("\nscan\n", outputs)

    # scan with transpose
    outputs_t = transpose_scan(ins)
    print("\nscan with transpose\n", outputs_t)