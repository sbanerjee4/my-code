import java.util.*;
import java.io.*;
import java.lang.*;
public class entertainment
{
   static ArrayList[] cows;
   static boolean[] marked;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("entertainment.in"));
      int n = in.nextInt(), m = in.nextInt();
      cows = new ArrayList[n];
      for(int i = 0; i < n; i++) cows[i] = new ArrayList<Edge>(); 
      for(int i = 0; i < m; i++)
      {
         char ch = in.next().charAt(0);
         int a = in.nextInt(), b = in.nextInt();
         cows[a - 1].add(new Edge(b - 1, ch));
         cows[b - 1].add(new Edge(a - 1, ch));
      
      }
      
      marked = new boolean[n];
      int ans = 0;
      int[] color = new int[n];
      for(int i = 0; i < n; i++)
      {
         if(!marked[i])
         {
            ArrayList<Integer> cc = new ArrayList<>();
            cc.add(i);
            ans++;
            color[i] = 1;
            ff(i, cc);
            if(!dfs(color, n, i))
            {
               System.out.println(0);
               System.exit(0);
            }
         }
      }
      
      System.out.print(1);
      for(int i = 0; i < ans; i++)
         System.out.print(0);
      System.out.println();
   }
   
   static void ff(int pos, ArrayList<Integer> cc)
   {
      for(int i = 0; i < cows[pos].size(); i++)
      {
         if(!marked[((Edge)cows[pos].get(i)).to])
         {
            marked[((Edge)cows[pos].get(i)).to] = true;
            cc.add(((Edge)cows[pos].get(i)).to);
            ff(((Edge)cows[pos].get(i)).to, cc);
         }
      }
   }
   
   static boolean dfs(int[] color, int num, int cow)
   {
      for(Object o : cows[cow])
      {
         Edge e = (Edge)o;
         if(e.ch == 'S')
         {
            if(color[e.to] != 0 && color[e.to] != num)
               return false;
            if(color[e.to] == 0)
            {
               color[e.to] = num;
               dfs(color, num, e.to);
            }
         }
         else
         {
            if(color[e.to] != 0 && color[e.to] == num)
               return false;
            if(color[e.to] == 0)
            {
               color[e.to] = 3 - num;
               dfs(color, 3 - num, e.to);
            }
         }
      }
      
      return true;          
   }
   
   public static class Edge
   {
      public int to;
      public char ch;
      public Edge(int a, char b)
      {
         to = a;
         ch = b;
      }
   }
}
