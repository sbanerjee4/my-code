import java.util.*;
import java.io.*;
import java.lang.*;
public class tunnels
{
   static int n, q;
   static ArrayList[] adj;
   static boolean[] visited;
   static boolean[] reach;
   static int k;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("tunnels.in"));
      n = in.nextInt();
      q = in.nextInt();
      adj = new ArrayList[n];
      visited = new boolean[n];
      for(int i = 0; i < n; i++) adj[i] = new ArrayList<Pair>();
      for(int i = 0; i < n - 1; i++)
      {
         int a = in.nextInt() - 1, b = in.nextInt() - 1;
         long c = in.nextLong();
         adj[a].add(new Pair(b, c));
         adj[b].add(new Pair(a, c));
      }
      
      for(int i = 0; i < q; i++)
      {
         k = in.nextInt();
         int temp = in.nextInt() - 1;
         visited = new boolean[n];
         reach = new boolean[n];
         dfs(temp, 0);
         int count = 0;
         for(int j = 0; j < visited.length; j++)
            if(reach[j]) count++;
         System.out.println(count);
      }
   }
   
   static void dfs(int nest, long len)
   {
      if(visited[nest]) 
         return;
      if(len >= k) reach[nest] = true;
      visited[nest] = true;
      for(Object p : adj[nest])
         if(len == 0)
            dfs(((Pair)p).nest, ((Pair)p).weight);
         else
            dfs(((Pair)p).nest, Math.min(len, ((Pair)p).weight));
   }
   
   static class Pair
   {
      public int nest;
      public long weight;
      public Pair(int n, long w)
      {
         nest = n;
         weight = w;
      }
   }
}