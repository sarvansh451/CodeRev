import java.util.*;

public class DijkstraPQ {
    static final int INF = Integer.MAX_VALUE;

    public static void main(String[] args) {
        int n = 5; // number of nodes

        List<List<int[]>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }

        // Add edges: {node, weight}
        adj.get(0).add(new int[]{1, 2});
        adj.get(1).add(new int[]{0, 2});

        adj.get(0).add(new int[]{2, 4});
        adj.get(2).add(new int[]{0, 4});

        adj.get(1).add(new int[]{2, 1});
        adj.get(2).add(new int[]{1, 1});

        adj.get(1).add(new int[]{3, 7});
        adj.get(3).add(new int[]{1, 7});

        adj.get(2).add(new int[]{4, 3});
        adj.get(4).add(new int[]{2, 3});

        adj.get(3).add(new int[]{4, 1});
        adj.get(4).add(new int[]{3, 1});

        int source = 0;
        int[] dist = new int[n];
        Arrays.fill(dist, INF);
        dist[source] = 0;

        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return Integer.compare(a[0], b[0]);
            }
        });

        pq.offer(new int[]{0, source}); // {distance, node}

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int d = curr[0];
            int u = curr[1];

            if (d > dist[u]) continue;

            List<int[]> neighbors = adj.get(u);
            for (int i = 0; i < neighbors.size(); i++) {
                int[] edge = neighbors.get(i);
                int v = edge[0];
                int w = edge[1];

                if (dist[v] > d + w) {
                    dist[v] = d + w;
                    pq.offer(new int[]{dist[v], v});
                }
            }
        }

        System.out.println("Distances from source:");
        for (int i = 0; i < n; i++) {
            System.out.println("Node " + i + ": " + dist[i]);
        }
    }
}
