import java.util.*;
import java.io.*;
import java.lang.*;

public class hibernation
{
   private static int[] a;
   private static int lim;
   private static int n;
   private static int best;
   private static int sum;
   
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("hibernation.in"));
      lim = in.nextInt();
      n = in.nextInt();
      a = new int[n];
      sum = 0;
      best = 0;
      for(int i = 0; i < n; i++)
         a[i] = in.nextInt();
      in.close();
       
      solve(0, 0);
      
      System.out.println(best);
   }
   
   public static void solve(int pos, int sum)
   {
      if(sum >= lim) return;
      if(pos == n)
      {
         if(sum <  lim && sum > best)
            best = sum;
         
         return;
      }
      solve(pos + 1, sum);
      solve(pos + 1, sum + a[pos]);
   }
}