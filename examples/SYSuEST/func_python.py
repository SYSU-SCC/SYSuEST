def check_solved(phase, a, N):
    from fractions import Fraction
    from math import gcd

    r = Fraction(phase).limit_denominator(N).denominator
    print(f"Find period r = {r}")
    guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
    for guess in guesses:
        if guess not in [1, N] and (N%guess)==0:
            print(f"Congratulations! Find factor {guess}!")
            print(f"{N} = {guess} x {N//guess}\n")
            return 1
    print("Factorization failed!")
    return 0
