import java.io.*;
import java.util.*;
import java.lang.*;
public class watertank
{
   static boolean[][] visited;
   static long[] f;
   static long[] s;
   static long[] d;
   static int n, b;
   static long ans = 0;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("watertank.in"));
      n = in.nextInt();
      b = in.nextInt();
      f = new long[n];
      s = new long[b + 1];
      d = new long[b + 1];
      for(int i = 0; i < n; i++) f[i] = in.nextLong();
    
      for(int i = 1; i <= b; i++)
      {
        s[i] = in.nextLong();
        d[i] = in.nextLong();
      }
      
      visited = new boolean[n][b];
      long ans = dfs(0, 0);
      System.out.println(ans - 1);
   }
   
   static long dfs(int x, int y)
   {
      if(visited[x][y]) 
         return ans;
      if(x == n - 1) 
         return y;
      ans = Integer.MAX_VALUE - n;
      visited[x][y] = true;
        
      for(int i = y + 1; i < b; i++)
         if(s[i] >= f[x])
            ans = Math.min(ans, dfs(x, i));
      for(int i = x + 1; i <= Math.min(x + d[y], n - 1); i++)
         if(s[y] >= f[i])
            ans = Math.min(ans, dfs(i, y));
      return ans;
   }
   
   static class Boot
   {
      public long s;
      public long d;
      public Boot(long a, long b)
      {
         s = a;
         d = b;
      }
   }
}