import java.util.*;
import java.io.*;
import java.lang.*;
public class vacation
{
   static int k, n, m;
   static int[] cow;
   static ArrayList[] a;
   static boolean[] visited;
   static int[] frequency;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("vacation.in"));
      k = in.nextInt();
      n = in.nextInt();
      m = in.nextInt();
      cow = new int[k];
      a = new ArrayList[m];
      for(int i = 0; i < m; i++) a[i] = new ArrayList<Integer>();
      for(int i = 0; i < k; i++)
         cow[i] = in.nextInt() - 1;
      for(int i = 0; i < m; i++)
         a[in.nextInt() - 1].add(in.nextInt() - 1);
      in.close();
      
      visited = new boolean[n];
      frequency = new int[n];
      for(int i = 0; i < k; i++)
      {
         visited = new boolean[n];
         dfs(cow[i]);
         for(int j = 0; j < n; j++)
            if(visited[j]) frequency[j]++;  
      }
      
      int count = 0;
      for(int i = 0; i < frequency.length; i++)
         if(frequency[i] == k)
            count++;
      System.out.println(count);
   }
   
   public static void dfs(int pos)
   {
      visited[pos] = true;
      for(int i = 0; i < a[pos].size(); i++)
         if(!visited[(int)a[pos].get(i)])
            dfs((int)a[pos].get(i));
   }
}