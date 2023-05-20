import java.util.*;
import java.io.*;
import java.lang.*;
public class grass
{
   static int n, m;
   static ArrayList[] up;
   static ArrayList[] down;
   static int countup = 0, countdown = 0;
   static boolean[] visited;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("grass.in"));
      n = in.nextInt();
      m = in.nextInt();
      up = new ArrayList[n];
      for(int i = 0; i < n; i++)
         up[i] = new ArrayList<Integer>();
      down = new ArrayList[n];
      for(int i = 0; i < n; i++)
         down[i] = new ArrayList<Integer>();
      
      for(int i = 0; i < m; i++)
      {
         int t1 = in.nextInt(), t2 = in.nextInt();
         t1--;
         t2--;
         up[t1].add(t2);
         down[t2].add(t1);
      }
      in.close();
      
      int ans = 0;
      for(int i = 0; i < n; i++)
      {
         countdown = 0;
         countup = 0;
         visited = new boolean[n];
         visited[i] = true;
         dfsdown(i);
         dfsup(i);
         if(countdown + countup == n - 1) ans++;
      }
      
      System.out.println(ans);
   }
   
   public static void dfsup(int pos)
   {
      for(int i = 0; i < up[pos].size(); i++)
         if(!visited[(int)up[pos].get(i)])
         {
            countup++;
            visited[(int)up[pos].get(i)] = true;
            dfsup((int)up[pos].get(i));
         }
   }
   
   public static void dfsdown(int pos)
   {
      for(int i = 0; i < down[pos].size(); i++)
         if(!visited[(int)down[pos].get(i)])
         {
            countdown++;
            visited[(int)down[pos].get(i)] = true;
            dfsdown((int)down[pos].get(i));
         }
   }
}