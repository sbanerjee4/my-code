import java.util.*;
import java.io.*;
import java.lang.*;

public class wormsort
{
   static int n, m;
   static String cows = "";
   static wh[] whs;
   static HashSet<String> set = new HashSet<>();
   static String goal = "";
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("wormsort.in"));
      n = in.nextInt();
      m = in.nextInt();
      for(int i = 0; i < n; i++) goal += (i + 1) + "";
      for(int i = 0; i < n; i++) cows += in.nextInt() + "";
      whs = new wh[m];
      for(int i = 0; i < m; i++) whs[i] = new wh(in.nextInt() - 1, in.nextInt() - 1, in.nextLong());
      in.close();
      
      Arrays.sort(whs);
      long result = 0;
      if(cows.equals(goal)) result = -1;
      else result = upperbound();
      
      PrintWriter out = new PrintWriter(new File("wormsort.out"));
      System.out.println(result);
      out.println(result);
      out.close();
   }
   
   public static long upperbound()
   {
      int a = 0, b = whs.length - 1;
      while(a < b)
      {
         int mid = (a + b) / 2;
         if(dfs(cows, -1, Long.MAX_VALUE, whs[mid].w))
            a = mid + 1;
         else 
            b = mid;
      }
      
      return whs[a - 1].w;
   }
   
   static boolean dfs(String current, int recent, long min, long bsval)
   {
      if(current.equals(goal)) 
         return true;
      for(int i = pos(bsval); i < m; i++)
         if(i != recent && whs[i].w >= bsval)
         {
            String ts = swap(current, i);
            return dfs(ts, i, Math.min(min, whs[i].w), bsval);
         }
      return false;
   }
   
   static int pos(long bsval)
   {
      int a = 0, b = whs.length;
      while(a < b)
      {
         int mid = (a + b) / 2;
         if(bsval <= whs[mid].w)
            b = mid;
         else 
            a = mid + 1;
      }
      
      return a;
   }
   
   static String swap(String s, int pos)
   {
      int a = Math.min(whs[pos].a, whs[pos].b);
      int b = Math.max(whs[pos].a, whs[pos].b);
      return s.substring(0, a) + s.charAt(b) + s.substring(a + 1, b) + s.charAt(a) + s.substring(b + 1);
   }
   
   static class wh implements Comparable<wh>
   {
      public int a;
      public int b;
      public long w;
      public wh(int x, int y, long z)
      {
         a = x;
         b = y;
         w = z;
      }
      
      public int compareTo(wh temp)
      {
         return Long.compare(this.w, temp.w);    
      }
   }
}