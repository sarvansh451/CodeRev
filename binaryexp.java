import java.util.Scanner;

public class ModularInverse {
    
    // Binary exponentiation function
    static long pow(long a, long b, long m) {
        if (b == 0) return 1;
        if (b % 2 == 0) {
            long t = pow(a, b / 2, m);
            return (t * t) % m;
        } else {
            long t = pow(a, (b - 1) / 2, m);
            t = (t * t) % m;
            return (a * t) % m;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        long a = sc.nextLong();  // input a
        long b = sc.nextLong();  // input b
        long m = sc.nextLong();  // input mod value

        // Incorrect way (just for comparison, not valid if m is not divisible)
        long res1 = (a / b) % m;

        // Convert 'a' to mod m to keep it clean
        a = a % m;

        // Correct way using Fermat’s Little Theorem
        long inv_b = pow(b, m - 2, m);  // Modular inverse of b
        long res2 = (a * inv_b) % m;   // (a / b) % m using modular inverse

        System.out.println(res1 + " " + res2);
    }
}
