import java.io.*;
import java.util.*;
import java.lang.*;
public class book
{
   static int min = Integer.MAX_VALUE;
   static int[] a;
   static int sum1 = 0;
   static int sum2 = 0;
   static int sum3 = 0;
   static int n;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("book.in"));
      n = in.nextInt();
      a = new int[n];
      for(int i = 0; i < n; i++)
         a[i] = in.nextInt();
      
      in.close();
      
      Arrays.sort(a);
      sum1 = a[0];
      dfs(1);
      System.out.println(min);
   }
   
   public static void dfs(int pos)
   {
      if(pos == n)
      {
         min = Math.min(min, Math.max(Math.max(sum1, sum2), sum3));
         return;
      }
      
      for(int i = 1; i <= 3; i++)
      {
         if(i == 1 && sum1 + a[pos] < min)
         {
            sum1 += a[pos];
            dfs(pos + 1);
            sum1 -= a[pos];
         }
         else if(i == 2 && sum2 + a[pos] < min)
         {
            sum2 += a[pos];
            dfs(pos + 1);
            sum2 -= a[pos];
         }
         else if(i == 3 && sum3 + a[pos] < min)
         {
            sum3 += a[pos];
            dfs(pos + 1);
            sum3 -= a[pos];
         }
      }
   }
   
   public static int biggest(String s)
   {
      int[] b = new int[3];
      for(int i = 0; i < s.length(); i++)
         b[Integer.parseInt(s.substring(i, i + 1))] += a[i];
      Arrays.sort(b);
      return b[2];
   }
}