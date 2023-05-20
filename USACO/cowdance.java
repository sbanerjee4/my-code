import java.util.*;
import java.io.*;
import java.lang.*;

public class cowdance
{
   static int n;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("cowdance.in"));
      n = in.nextInt();
      int t = in.nextInt();
      int[] d = new int[n];
      for(int i = 0; i < n; i++)
         d[i] = in.nextInt();
      in.close();
     
      int result = lowerbound(d, t);
     
      PrintWriter out = new PrintWriter(new File("cowdance.out"));
      out.println(result);
      out.close();
      System.out.println(result);
   }
   
   public static int lowerbound(int[] d, int t)
   {
      int a = 1, b = d.length;
      while(a < b)
      {
         int mid = (a + b) / 2;
         if(time(d, mid) <= t)
            b = mid;
         else 
            a = mid + 1;
      }
      
      return a;
   }
   
   public static long time(int[] d, int cows)
   {
      PriorityQueue<Long> set = new PriorityQueue<>();
      for(int i = 0; i < cows; i++)
         set.add((long)d[i]);
      for(int i = cows; i < d.length; i++)
      {
         long cur = (long)set.peek();
         set.remove(set.peek());
         set.add(cur + d[i]);
      }
        
      long result = set.peek();
      set.remove(set.peek());
      while(set.size() > 0)
      {
         result = set.peek();
         set.remove(set.peek());
      }
      
      return result;
   }
}