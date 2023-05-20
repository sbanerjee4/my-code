import java.util.*;
import java.io.*;
import java.lang.*;

public class scout
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("scout.in"));
      int k = in.nextInt();
      int l = in.nextInt();
      int[] knots = new int[k];
      boolean[] rope = new boolean[2 * l + 1];
   
      for(int i = 0; i < knots.length; i++)
      {
         knots[i] = in.nextInt() * 2;
         rope[knots[i]] = true;
      }
      
      Arrays.sort(knots);
      // rope[rope.length - 1] = true;
      
      in.close();
      
      int result = 0;
      
      for(int i = 1; i < knots.length - 1; i++)
         if(good(knots[i], rope))
           result++;
      
      for(int i = 1; i < knots.length; i++)
      {
         int temp = (knots[i] + knots[i - 1]) / 2;
         if(good(temp, rope))
            result++;
      }
      
      System.out.println(result);
   }
   
   public static boolean good(int pos, boolean[] a)
   {
      int left = pos, right = pos;
      while(left >= 0 && right < a.length)
      {
         if(a[left] != a[right]) 
            return false;
         left--;
         right++;
      }
      return true; 
   }
}