import java.io.*;
import java.util.*;
import java.lang.*;
public class minesweeper
{
   static int[][] a;
   static int[][] work;
   static int n, m, k;
   static TreeSet<Pair> set = new TreeSet<>();
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("minesweeper.in"));
      n = in.nextInt();
      m = in.nextInt();
      k = in.nextInt();
      a = new int[n][m];
      work = new int[n][m];
      for(int i = 0; i < n; i++)
         for(int j = 0; j < m; j++)
            a[i][j] = in.nextInt();
      for(int i = 0; i < n; i++)
         for(int j = 0; j < m; j++)
            dfs(i, j, 0); 
   }
   
   public static void dfs(int i, int j, int usedMines)
   {
      if(a[i][j] == 0) 
         return;
      if(usedMines > k) 
         return;
      if(usedMines == k)
      {
         for(int t = 0; t < n; t++)
            for(int s = 0; s < m; s++)
               if(a[t][s] != work[t][s]) 
                  return;
         print();
         System.exit(0);           
      }
      
      if(violate(i, j)) 
      {
         if(j < m - 1) dfs(i, j + 1, usedMines);
         else if(i < n - 1) dfs(i + 1, 0, usedMines);
      }
      else
      {
         work[i][j]++;
         
         int[] di = {-1, -1, -1, 0, 0, 1, 1, 1};
         int[] dj = {-1, 0, 1, -1, 1, -1, 0, 1};
         for(int t = 0; t < 8; t++)
            if(i + di[t] >= 0 && i + di[t] < n && j + dj[t] >= 0 && j + dj[t] < m)
               work[i + di[t]][j + dj[t]]++;
         
         set.add(new Pair(i, j));         
         if(j < m - 1) dfs(i, j + 1, usedMines + 1);
         else if(i < n - 1) dfs(i + 1, 0, usedMines + 1);
         set.remove(new Pair(i, j)); 
         
         work[i][j]--;
         for(int t = 0; t < 8; t++)
            if(i + di[t] >= 0 && i + di[t] < n && j + dj[t] >= 0 && j + dj[t] < m)
               work[i + di[t]][j + dj[t]]--;
         
         if(j < m - 1) dfs(i, j + 1, usedMines);
         else if(i < n - 1) dfs(i + 1, 0, usedMines);
      }
   }
   
   public static boolean violate(int i, int j)
   {
      if(i < 0 || i >= n || j < 0 || j >= m) 
         return true;
      if(work[i][j] + 1 > a[i][j]) 
         return true;
      if(i > 0 && j > 0)
         if(a[i - 1][j - 1] + 1 != a[i][j]) 
            return true;
      int[] di = {-1, -1, -1, 0, 0, 1, 1, 1};
      int[] dj = {-1, 0, 1, -1, 1, -1, 0, 1};
      for(int t = 0; t < 8; t++)
         if(i + di[t] >= 0 && i + di[t] < n && j + dj[t] >= 0 && j + dj[t] < m)
            if(work[i + di[t]][j + dj[t]] + 1 > a[i + di[t]][j + dj[t]]) 
               return true;
      return false;
   }
   
   public static void print()
   {
      for(Pair s : set) System.out.println(s); 
   }
   
   static class Pair implements Comparable<Pair>
   {
      public int x;
      public int y;
      public Pair(int a, int b)
      {
         x = a;
         y = b;
      }
      public int compareTo(Pair p)
      {
         if(x != p.x) 
            return Integer.compare(x, p.x);
         return Integer.compare(y, p.y);
      }
      public String toString()
      {
         return (x + 1) + " " + (y + 1);
      }
   }
}