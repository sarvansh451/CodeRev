import java.util.*;

public class HashMapTraversal {
    public static void main(String[] args) {
        HashMap<String, Integer> map = new HashMap<>();
        map.put("apple", 3);
        map.put("banana", 5);
        map.put("orange", 2);

        // 1. Traversal using keySet()
        System.out.println("Traversal using keySet():");
        for (String key : map.keySet()) {
            System.out.println("Key: " + key + ", Value: " + map.get(key));
        }

        // 2. Traversal using values()
        System.out.println("\nTraversal using values():");
        for (Integer val : map.values()) {
            System.out.println("Value: " + val);
        }

        // 3. Traversal using entrySet()
        System.out.println("\nTraversal using entrySet():");
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue());
        }
    }
}
