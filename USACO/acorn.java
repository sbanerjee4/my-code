import java.util.*;
import java.io.*;
import java.lang.*;

public class acorn
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("acorn.in"));
      int n = in.nextInt();
      int[] a = new int[n + 1];
      int length = 0;
      int keep = 0;
      int num = 0;
      int best = 0;
      
      for(int i = 1; i <= n; i++)
         a[i] = in.nextInt();
      
      int[] rev = new int[n + 1];
      for(int i = 1; i <= n; i++)
         rev[in.nextInt()] = i;

      
      in.close();

      for(int i = 1; i <= n; i++)
      {
         length = 0;
         keep = a[i];
         a[i] = 0;
         if(rev[keep] != 0)
         {
            if(rev[keep] != i) num++;
            while(rev[keep] != 0)
            {
               a[i] = 0;
               swap(rev[keep], i, a);
               keep = a[i];
               length++;
            }
            rev[i] = 0;
         }
         
         best = Math.max(best, length);
      }
      
      if(num == 0) best = -1;
      System.out.println(num + " " + best);
   }
   
   public static void swap(int x, int y, int[] a)
   {
      int temp = a[x];
      a[x] = a[y];
      a[y] = temp;
   }
}