
import numpy as np
from signal_flow_game.polar_codes import PolarCoDec

def run_test():
    np.random.seed(42)
    N = 256
    K = 128
    pc = PolarCoDec(N, K)
    
    # Message
    msg = np.random.randint(0, 2, K)
    enc = pc.encode(msg)
    
    # BPSK modulation: 0->1, 1->-1
    bpsk = 1.0 - 2.0 * enc
    
    # Add noise for -4dB
    # SNR = 10log(Es/N0) = -4.1
    # Es = 1.
    # N0 = Es / 10^(-0.41) = 1 / 0.389 = 2.57
    # Noise Variance sigma^2 = N0/2 = 1.285
    sigma = np.sqrt(1.285)
    
    noise = sigma * np.random.randn(N)
    rx = bpsk + noise
    
    # LLR = 2 * y / sigma^2
    llr = 2.0 * rx / (sigma**2)
    
    # SC
    u_sc = pc.decode(llr, 'SC')
    ber_sc = np.mean(msg != u_sc)
    
    # SCL L=8
    u_scl = pc.decode(llr, 'SCL', list_size=8)
    ber_scl = np.mean(msg != u_scl)

    # BP 20 iters
    u_bp = pc.decode(llr, 'BP', max_iter=20)
    ber_bp = np.mean(msg != u_bp)
    
    print(f"SNR: -4.1dB")
    print(f"SC BER: {ber_sc}")
    print(f"SCL BER: {ber_scl}")
    print(f"BP BER: {ber_bp}")
    
    if ber_sc == ber_scl and ber_sc == ber_bp:
        print("ALL SC/SCL/BP ARE IDENTICAL!")
    else:
        print("Differences found.")

if __name__ == "__main__":
    run_test()
