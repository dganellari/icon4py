import jax
import jax.numpy as jnp
from jax import lax
import jax.export as export

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
    with open('scan_output.txt', 'w') as f:
        f.write(str(outputs))

    # scan with transpose
    outputs_t = transpose_scan(ins)
    with open('transpose_scan_output.txt', 'w') as f:
        f.write(str(outputs_t))

    # Lower and compile scan
    lowered_scan = jax.jit(scan).lower(ins)
    compiled_scan = lowered_scan.compile()
    with open('scan_compiled_text.txt', 'w') as f:
        f.write(compiled_scan.as_text())

    # Lower and compile transpose_scan
    lowered_transpose_scan = jax.jit(transpose_scan).lower(ins)
    compiled_transpose_scan = lowered_transpose_scan.compile()
    with open('transpose_scan_compiled_text.txt', 'w') as f:
        f.write(compiled_transpose_scan.as_text())