import java.util.*;

public class Solution {
    public int getMinTime(int[] taskMemory, int[] taskType, int maxMemory) {
        // Step 1: Group tasks by their type
        Map<Integer, List<Integer>> tasksByType = new HashMap<>();
        for (int i = 0; i < taskMemory.length; i++) {
            tasksByType.putIfAbsent(taskType[i], new ArrayList<>());
            tasksByType.get(taskType[i]).add(taskMemory[i]);
        }

        int totalTime = 0;

        // Step 2: For each type, sort and try to pair tasks
        for (List<Integer> memoryList : tasksByType.values()) {
            Collections.sort(memoryList); // sort ascending
            int start = 0;
            int end = memoryList.size() - 1;

            while (start < end) {
                if (memoryList.get(start) + memoryList.get(end) <= maxMemory) {
                    // can run together
                    totalTime++;
                    start++;
                    end--;
                } else {
                    // run largest alone
                    totalTime++;
                    end--;
                }
            }

            // Step 3: If one task left unpaired
            if (start == end) {
                totalTime++;
            }
        }

        return totalTime;
    }
}
