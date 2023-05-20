import java.io.*;
import java.util.*;
import java.lang.*;
public class light
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("light.in"));
      int[] a = new int[20];
      for(int i = 0; i < 20; i++)
         a[i] = in.nextInt();
      in.close();
      
      int best = Integer.MAX_VALUE;
      for(int i = 0; i < (1 << 20); i++)
      {
         int sum = 0;
         int[] temp = a.clone();
         for(int j = 0; j < 20; j++)
         {
            if((i & (1 << j)) >0)
            {
               int x = j - 1;
               int y = j + 1;
               if(x >= 0) toggle(x, temp);
               toggle(j, temp);
               if(y < 20) toggle(y, temp);
               sum++;
            }            
         }
         
         boolean hasOne = false;
         
         for(int j = 0; j < temp.length; j++)
            if(temp[j] == 1) hasOne = true;
         if(!hasOne) best = Math.min(best, sum);
      }
      
      System.out.println(best);
   }
   
   public static void toggle(int pos, int[] a)
   {
      if(a[pos] == 0) a[pos] = 1;
      else a[pos] = 0;   
   }
}