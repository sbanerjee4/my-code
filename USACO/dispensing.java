import java.io.*;
import java.util.*;
import java.lang.*;
public class dispensing
{
   static int n, m;
   static ArrayList[][] a;
   static boolean[][] vis;
   static boolean[][] on;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("dispensing.in"));
      n = in.nextInt();
      m = in.nextInt();
      a = new ArrayList[n][m];
      
      vis = new boolean[n][n];
      on = new boolean[n][n];
      for(int i = 0; i < n; i++)
         for(int j = 0; j < n; j++)
            a[i][j] = new ArrayList<Pair>();
      
      for(int i = 0; i < m; i++)
         a[in.nextInt() - 1][in.nextInt() - 1].add(new Pair(in.nextInt() - 1, in.nextInt() - 1));
      
      in.close();
      
      dfs(1, 1);
   }
   
   public static void dfs(int i, int j)
   {
      vis[i][j] = true;
      for(Pair s : a[i][j])
         vis[s.x][s.y] = true;
      
      int[] di = {0, 0, 1, -1};
      int[] dj = {1, -1, 0, 0};
      
      for(Pair s : a[i][j])
         for(int t = 0; t < 4; t++)
            if(s.x + di[i]
   }
   
   static class Pair
   {
      public int x;
      public int y;
      public Pair(int a, int b)
      {
         x = a;
         y = b;
      }
   }
}