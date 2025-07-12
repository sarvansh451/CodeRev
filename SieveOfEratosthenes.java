import java.util.*;

public class SieveOfEratosthenes {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter n to find all primes up to n: ");
        int n = sc.nextInt();

        // Step 1: Create boolean array, initially true
        boolean[] isPrime = new boolean[n + 1];
        Arrays.fill(isPrime, true);  // assume all numbers are prime
        isPrime[0] = false; // 0 is not prime
        isPrime[1] = false; // 1 is not prime

        // Step 2: Sieve algorithm
        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                // Mark all multiples of i as not prime
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false;
                }
            }
        }

        // Step 3: Print all prime numbers
        System.out.println("Prime numbers up to " + n + ":");
        for (int i = 2; i <= n; i++) {
            if (isPrime[i]) {
                System.out.print(i + " ");
            }
        }
    }
}
