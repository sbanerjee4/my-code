import java.util.*;
import java.io.*;
import java.lang.*;

public class rabbit
{
   private static long[] rabbitx;
   private static long[] rabbity;
   private static long[] seatx;
   private static long[] seaty;
   
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("rabbit.in"));
      int n = in.nextInt(), m = in.nextInt();
      rabbitx = new long[n + 1];
      rabbity = new long[n + 1];
      long[] marked = new long[n + 1];
      long mini = Long.MAX_VALUE;
      int pos = 0;
      for(int i = 1; i <= n; i++)
      {
         rabbitx[i] = in.nextLong();
         rabbity[i] = in.nextLong();
      }
      
      seatx = new long[m + 1];
      seaty = new long[m + 1];
      for(int i = 1; i <= m; i++)
      {
         seatx[i] = in.nextLong();
         seaty[i] = in.nextLong();
      }
      
      in.close();
      
      for(int i = 1; i <= m; i++)
      {
         mini = Long.MAX_VALUE;
         pos = 0;
         for(int j = 1; j <= n; j++)
         {
            if(marked[j] == 0)
            {
               long d = (long)dist(i, j);
               if(d < mini)
               {
                  mini = d;
                  pos = j;
               }
            }
         }
         
         marked[pos] = 1;
      }
      
      int count = 0;
      for(int i = 1; i <= n; i++)
         if(marked[i] == 0)
         {
            System.out.println(i);
            count++;
         }
      
      if(count == 0) System.out.println(0);
   }
   
   public static double dist(int y, int x)
   {
      return Math.pow(rabbitx[x] - seatx[y], 2) + Math.pow(rabbity[x] - seaty[y], 2);
   }
}