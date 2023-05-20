import java.util.*;
import java.io.*;
import java.lang.*;

public class tourism
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("tourism.in"));
      int n = in.nextInt();
      int[] x = new int[n + 1];
      int[] y = new int[n + 1];
      for(int i = 1; i <= n; i++)
      {
         x[i] = in.nextInt();
         y[i] = in.nextInt();
      }
      
      in.close();
      
      int td = 0;
      for(int i = 1; i < n; i++)
         td += Math.abs(x[i] - x[i + 1]) + Math.abs(y[i] - y[i + 1]);
         
      int ans = Integer.MAX_VALUE, temp = 0;
      for(int i = 2; i < n; i++)
      {
         temp = td;
         temp -= Math.abs(x[i] - x[i - 1]) + Math.abs(y[i] - y[i - 1]);
         temp -= Math.abs(x[i] - x[i + 1]) + Math.abs(y[i] - y[i + 1]);
         temp += Math.abs(x[i - 1] - x[i + 1]) + Math.abs(y[i - 1] - y[i + 1]);
         ans = Math.min(ans, temp);
      }  
      
      System.out.println(ans);
   }
}