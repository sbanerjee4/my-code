import java.util.*;
import java.io.*;
public class teleporters
{
   static int n, m;
   static int[] cows;
   static int[] rev;
   static ArrayList[] adj;
   static boolean[] visited;
   static HashSet<Integer> s1;
   static HashSet<Integer> s2;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("teleporters.in"));
      n = in.nextInt();
      m = in.nextInt();
      long r = 0;
      cows = new int[n];
      rev = new int[n];
      s1 = new HashSet<>();
      s2 = new HashSet<>();
      for(int i = 0; i < cows.length; i++)
      {
         cows[i] = in.nextInt() - 1;
         rev[cows[i]] = i;
      }
      
      if(sorted())
      {
         System.out.println(-1);
         System.exit(0);
      }
      
      adj = new ArrayList[n];
      for(int i = 0; i < n; i++) adj[i] = new ArrayList<Pair>();
      for(int i = 0; i < m; i++)
      {
         int a = in.nextInt() - 1, b = in.nextInt() - 1;
         long c = in.nextLong();
         adj[a].add(new Pair(b, c));
         adj[b].add(new Pair(a, c));
         r = Math.max(r, c);
      }
      
      in.close();
      
      long l = 0, mid;
      while(l < r)
      {
         mid = (l + r + 1) / 2;
         if(isok(mid)) l = mid;
         else r = mid - 1;
      }
      
      System.out.println(l);
   }
   
   static boolean sorted()
   {
      for(int i = 1; i < cows.length; i++)
         if(cows[i] < cows[i - 1]) 
            return false;
      return true;
   }
   
   static boolean isok(long mid)
   {
      visited = new boolean[n];
      for(int i = 0; i < n; i++)
      {
         s1.clear();
         s2.clear();
         if(!visited[i])
         {
            s1.add(cows[i]);
            s2.add(rev[i]);
            dfs(i, 0, mid);
            
            visited[i] = true;
         }
         
         if(!same())
            return false;
      }
      
      return true;
   }
   
   public static boolean same()
   {
      for(int l : s1)
         if(!s2.contains(l)) 
            return false;
      return true;
   }
   
   static void dfs(int tel, long len, long k)
   {
      s1.add(cows[tel]);
      s2.add(rev[tel]);
      visited[tel] = true;
      for(Pair p : (ArrayList<Pair>)adj[tel])
         if(!visited[p.tel])
         {
            if(len == 0 && p.height >= k)
               dfs(p.tel, p.height, k);
            else if(Math.min(len, p.height) >= k)
               dfs(p.tel, Math.min(len, p.height), k);
         }
   }
}
class Pair
{
   int tel;
   long height;
   Pair(int t, long h)
   {
      tel = t;
      height = h;
   }
}